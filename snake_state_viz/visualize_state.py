import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Duration
from std_msgs.msg import String, Header, ColorRGBA
# from snakelib_control.pykdl_utils.kdl_kinematics import KDLKinematics
from snakelib_control.pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from snakelib_msgs.msg import HebiSensors
from geometry_msgs.msg import Point

from urdf_parser_py.urdf import Robot

import PyKDL

import numpy as np
import pinocchio as pin

from tf2_ros import TransformBroadcaster, TransformStamped
from visualization_msgs.msg import Marker

from copy import deepcopy

def frame_to_tf(frame:PyKDL.Frame, frame_id:str, header:Header) -> TransformStamped: 
    # Extract translation
    translation = frame.p

    # Extract rotation as quaternion (x, y, z, w)
    q = frame.M.GetQuaternion()

    t_stamped = TransformStamped()
    t_stamped.header = header
    t_stamped.child_frame_id = frame_id
    
    # Set translation
    t_stamped.transform.translation.x = translation[0]
    t_stamped.transform.translation.y = translation[1]
    t_stamped.transform.translation.z = translation[2]

    # Set rotation
    t_stamped.transform.rotation.x = q[0]
    t_stamped.transform.rotation.y = q[1]
    t_stamped.transform.rotation.z = q[2]
    t_stamped.transform.rotation.w = q[3]

    return t_stamped

def pin_to_tf(frame:pin.SE3, frame_id:str, header:Header) -> TransformStamped:
    q = pin.Quaternion(frame.rotation)

    t_stamped = TransformStamped()
    t_stamped.header = header
    t_stamped.child_frame_id = frame_id

    # Set translation
    t_stamped.transform.translation.x = float(frame.translation[0])
    t_stamped.transform.translation.y = float(frame.translation[1])
    t_stamped.transform.translation.z = float(frame.translation[2])

    # Set rotation
    t_stamped.transform.rotation.x = float(q.x)
    t_stamped.transform.rotation.y = float(q.y)
    t_stamped.transform.rotation.z = float(q.z)
    t_stamped.transform.rotation.w = float(q.w)

    return t_stamped


def get_fk_all_frames(chain:PyKDL.Chain, joint_angles:list[float]) -> list[PyKDL.Frame]:
    """
    Computes forward kinematics for every segment/frame in a PyKDL.Chain.
    
    :param chain: PyKDL.Chain object containing the kinematic segments.
    :param joint_angles: list containing positions for all movable joints.
    :return: A list of PyKDL.Frame objects, one for each segment in the chain.
    """
    # Verify that the input joint array matches the chain's joint count
    if chain.getNrOfJoints() != len(joint_angles):
        raise ValueError("Joint angle count does not match the chain configuration.")
        
    frames_list = []
    
    # Start with an Identity Frame representing the robot base
    current_frame = PyKDL.Frame.Identity()
    
    joint_index = 0
    num_segments = chain.getNrOfSegments()
    
    # Iterate sequentially through every segment in the chain
    for i in range(num_segments):
        segment = chain.getSegment(i)
        joint = segment.getJoint()
        
        # Determine joint displacement depending on whether it's movable or fixed
        if joint.getTypeName() == "Fixed":
            joint_pos = 0.0
        else:
            try:
                joint_pos = joint_angles[joint_index]
            except IndexError as e:
                print(joint_index, len(joint_angles))
                raise e
            joint_index += 1
            
        # Get the relative transformation across this segment for the given joint state
        # Then multiply by the previous frame to accumulate the transform in the base frame
        current_frame = current_frame * segment.pose(joint_pos)
        
        # # Save a copy of the base-to-tip frame
        # tip_frame = current_frame * segment.getFrameToTip()
        # frames_list.append(PyKDL.Frame(tip_frame))
        frames_list.append(PyKDL.Frame(current_frame))

    frames_list.append(current_frame * segment.getFrameToTip())
    
    return frames_list[1:]

def mean_of_rotations(rotations, max_iters=10, tol=1e-6):
    """
    Computes the geometric mean of a list of pinocchio.SE3 matrices
    (or flat SO(3) rotations via an isolated free-flyer model joint type).
    """
    # Start with the first rotation as our initial guess
    mean_R = rotations[0]

    for _ in range(max_iters):
        # Compute the velocity vectors (tangent space error vectors) from mean to each rotation
        tangent_errors = []
        for R in rotations:
            # pin.log3 computes the 3D angular error vector (Lie algebra)
            error_vector = pin.log3(mean_R.T @ R)
            tangent_errors.append(error_vector)

        # Average the vectors in the flat tangent space
        mean_error = np.mean(tangent_errors, axis=0)

        # If the update is negligible, we've converged
        if np.linalg.norm(mean_error) < tol:
            break

        # Step towards the mean using the exponential map
        mean_R = mean_R @ pin.exp3(mean_error)

    return mean_R


class RobotOrientationEKF:
    def __init__(self, model, IMU_frame_name, dt=0.001):
        """
        An Extended Kalman Filter to estimate robot floating-base orientation 
        using an IMU accelerometer and encoder joint angles (q).

        State vector x = [q_base, omega_bias] (dim: 3 for orientation + 3 for bias = 6)
        Note: For simplicity on SO(3), we track orientation error or use exponential coordinates.
        Here we track roll/pitch/yaw error state or full orientation via rotation matrix/quaternion.
        """
        self.model = model
        self.data = model.createData()
        self.imu_frame_id = model.getFrameId(IMU_frame_name)
        self.dt = dt

        # Gravity vector in world frame
        self.g_world = np.array([0, 0, -9.81])

        # State initialization: R_base (Rotation matrix), bias (gyro/accel drift)
        self.R_base = np.eye(3)
        self.bias_omega = np.zeros(3)

        # Covariance matrices
        self.P = np.eye(6) * 0.01  # State covariance
        self.Q = np.eye(6) * 0.001 # Process noise covariance
        self.R = np.eye(3) * 0.1   # Measurement noise covariance (accelerometer)

    def skew(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def predict(self, omega_measured):
        """
        Predict step using gyroscope measurement (angular velocity).
        """
        # Unbiased angular velocity
        omega = omega_measured - self.bias_omega

        # Exponential map for rotation matrix update
        theta = np.linalg.norm(omega) * self.dt
        if theta > 1e-6:
            axis = omega / np.linalg.norm(omega)
            R_update = pin.exp3(axis * theta)
        else:
            R_update = np.eye(3)

        self.R_base = self.R_base @ R_update

        # Error state Jacobian (F matrix)
        F = np.eye(6)
        F[0:3, 3:6] = -self.R_base * self.dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update(self, q_joints, acc_measured):
        """
        Update step using joint angles (q) and accelerometer measurement.
        Assumes quasi-static assumption (acceleration is dominated by gravity).
        """
        # Update Pinocchio kinematics with current base estimation
        # q_full consists of [base_pos(3), base_quat(4), joints(n)]
        q_full = np.zeros(self.model.nq)
        # Set base orientation in Pinocchio q
        quat = pin.Quaternion(self.R_base)
        q_full[3:7] = quat.coeffs()
        q_full[7:] = q_joints

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        # Get current IMU frame orientation relative to world
        R_imu = self.data.oMf[self.imu_frame_id].rotation

        # Expected gravity in the IMU frame
        acc_predicted = R_imu.T @ (-self.g_world)

        # Innovation/Measurement residual
        y = acc_measured - acc_predicted

        # Measurement Jacobian H
        # H maps state errors to acceleration innovations
        H = np.zeros((3, 6))
        # Derivative of R_imu.T @ -g with respect to base rotation error
        H[0:3, 0:3] = self.skew(acc_predicted)

        # Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State correction
        dx = K @ y

        # Apply orientation correction
        self.R_base = self.R_base @ pin.exp3(dx[0:3])
        self.bias_omega += dx[3:6]

        # Covariance update
        self.P = (np.eye(6) - K @ H) @ self.P

        return self.R_base


class MinimalPublisher(Node):
    # For KDL
    _chain: PyKDL.Chain | None = None
    _joint_map: dict[int, str] = {}
    _imu_offsets: dict[int, str] = {}
    # For pinocchio
    _joint_names: list[str] = []
    _model: pin.Model | None = None
    _data: pin.Data | None = None
    _imu_offsets: list[pin.SE3] | None = None
    _yaw_offsets: list[pin.Quaternion] | None = None

    def __init__(self):
        super().__init__('minimal_publisher')
        self.sensor_sub = self.create_subscription(
            HebiSensors,
            '/hebi_sensors/data',
            self.hebi_cb,
            1
        )

        # Critical: Define a QoS profile that matches the publisher
        qos_profile = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL  # <-- Essential line
        )

        self.robot_description_sub = self.create_subscription(
            String,
            '/robot_description',
            self.robot_description_cb,
            qos_profile
            
        )

        self._marker_pub = self.create_publisher(
            Marker,
            "/accelerations",
            100,
        )

        self._marker_msg = Marker()
        self._marker_msg.lifetime = Duration(sec=0, nanosec=500000000)
        self._marker_msg.header.frame_id = 'head'
        self._marker_msg.type = Marker.LINE_LIST
        self._marker_msg.id = 0
        self._marker_msg.action = Marker.ADD
        self._marker_msg.color.r = 1.0
        self._marker_msg.color.g = 1.0
        self._marker_msg.color.b = 1.0
        self._marker_msg.color.a = 1.0
        self._marker_msg.scale.x = 0.01
        self._marker_msg.scale.y = 0.01
        self._marker_msg.scale.z = 0.01

        # Initialize the transform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

    def hebi_cb_kdl(self, msg:HebiSensors):
        if self._chain is None:
            return
        if not self._joint_map:
            for i in range(self._chain.getNrOfSegments()):
                name = self._chain.getSegment(i).getName()
                if name in msg.name:
                    self._joint_map[name] = i+1
        header = msg.header
        header.frame_id = 'head_link'


        self._marker_msg.header = header
        self._marker_msg.points.clear()

        frames = get_fk_all_frames(self._chain, msg.position)
        for i in range(len(msg.name)):
            name = msg.name[i]
            frame = frames[i]
            self._tf_broadcaster.sendTransform(frame_to_tf(frame, f'/test/{name}', header))
            base = frame * self._imu_offsets[name]
            accel = base * PyKDL.Vector(
                msg.lin_acc.x[i] * 0.02,
                msg.lin_acc.y[i] * 0.02,
                msg.lin_acc.z[i] * 0.02
            )
            pointA = Point(x=base.p.x(), y=base.p.y(), z=base.p.z())
            pointB = Point(x=accel.x(), y=accel.y(), z=accel.z())
            self._marker_msg.points.append(pointA)
            self._marker_msg.points.append(pointB)

        self._marker_pub.publish(self._marker_msg)

    def imu_to_head_frame(self, imu_val):
        # This only works after pin.forwardKinematics and pin.updateFramePlacements
        head_var = []
        for i, (v, name) in enumerate(zip(imu_val, self._joint_names)):
            frame_id = self._model.getFrameId(f'{name}_imu')
            R_imu = self._data.oMf[frame_id].rotation
            head_var.append(R_imu @ v)
        return head_var


    def link_to_head_frame(self, imu_val):
        # This only works after pin.forwardKinematics and pin.updateFramePlacements
        head_var = []
        for i, (v, name) in enumerate(zip(imu_val, self._joint_names)):
            frame_id = self._model.getFrameId(f'{name}_link')
            R_imu = self._data.oMf[frame_id].rotation
            head_var.append(R_imu @ v)
        return head_var


    def get_gravity(self, lin_acc:np.array, outlier_deg:float=10):
        # Accelerations should be provided in the head frame

        # Reject lin_acc near zero
        min_norm = 1e-3
        norms =  np.linalg.norm(lin_acc, axis=1)
        rejected = np.argwhere(norms <= min_norm).flatten().tolist()
        accepted = np.argwhere(norms > min_norm).flatten().tolist()
        U = np.array([lin_acc[i] for i in range(len(lin_acc)) if i in accepted])

        # Robust central direction: component-wise median of the unit vectors,
        # renormalized. Dominated by the agreeing majority, so outliers don't move it.
        med = np.median(U, axis=0)
        mn = np.linalg.norm(med)
        if mn < 1e-9: # degenerate median -> fall back
            med = U.mean(axis=0); mn = np.linalg.norm(med)
            if mn < 1e-9:
                return None, 0.0, [i for i in range(len(len(lin_acc)))]
        med = med / mn

        # Reject modules whose direction is > outlier_deg from the median.
        cos_thr = np.cos(np.radians(outlier_deg))
        keep = (U @ med) >= cos_thr
        if int(keep.sum()) < 4: # too few survivors -> keep all
            keep = np.ones(len(U), dtype=bool)
        rejected += [accepted[j] for j in range(len(accepted)) if not keep[j]]

        mean_vec = U[keep].mean(axis=0)
        agreement = float(np.linalg.norm(mean_vec)) # over survivors
        if agreement < 1e-6:
            return None, 0.0, rejected

        return mean_vec / agreement, agreement, rejected



    def hebi_cb(self, msg:HebiSensors):
        if self._model is None:
            return
        header = msg.header
        header.frame_id = 'head_link'
        self._marker_msg.header = header
        self._marker_msg.points.clear()
        self._marker_msg.colors.clear()
        q = np.array(msg.position, dtype=np.float64)
        v = np.array(msg.velocity, dtype=np.float64)
        pin.forwardKinematics(self._model, self._data, q, v)
        # pin.forwardKinematics(self._model, self._data, np.zeros_like(q), np.zeros_like(v))
        pin.updateFramePlacements(self._model, self._data)

        # Get orientations
        orientations = zip(
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        )
        orientations = [pin.Quaternion(*o).toRotationMatrix() for o in orientations]
        # orientations = self.imu_to_head_frame(orientations)
        # orientations = [b.rotation.T @ a for a, b in zip(orientations, self._imu_offsets)]
        # orientations = [pin.rpy.rpyToMatrix(np.pi/2, 0, np.pi) @ o for o in orientations]
        if self._yaw_offsets is None:
            self._yaw_offsets = [None] * len(orientations)

        for i, (mat, name) in enumerate(zip(orientations, self._joint_names)):
            frame_id = self._model.getFrameId(f'{name}_link')
            R = self._data.oMf[frame_id].rotation
            orientations[i] = self._imu_offsets[i].rotation.T @ orientations[i] @ pin.rpy.rpyToMatrix(0, np.pi/2, 0)
            orientations[i] = pin.rpy.rpyToMatrix(np.pi/2, 0, np.pi) @ orientations[i]
            orientations[i] = R  @ self._imu_offsets[i].rotation.T @ orientations[i]

            if self._yaw_offsets[i] is None:
                x_R = [1, 0, 0]
                # x_R = R[:, 0].flatten()
                z_O = orientations[i][:,2]
                # Compute the perpendicular component
                y_O = np.cross(x_R, z_O)
                y_O = y_O / np.linalg.norm(y_O)
                x_O = np.cross(z_O, y_O)
                x_O = x_O / np.linalg.norm(x_O)
                if np.dot(x_R, x_O) < 0:
                    x_O = -x_O
                new_O = np.vstack([x_O, y_O, z_O]).T
                if np.linalg.det(new_O) < 0:
                    y_O = -y_O
                new_O = np.vstack([x_O, y_O, z_O]).T
                self._yaw_offsets[i] = (new_O.T @ orientations[i]).T

            orientations[i] = orientations[i] @ self._yaw_offsets[i]

        assert np.allclose([np.linalg.det(o) for o in orientations], 1)

        world = pin.SE3(mean_of_rotations(orientations).T, np.array([0,0,0], np.float64))
        header_world = deepcopy(header)
        header_world.frame_id = 'world'
        self._tf_broadcaster.sendTransform(pin_to_tf(world, '/head_link', header_world))

        xs = [o[:,0] * 0.1 for o in orientations[0:]]
        ys = [o[:,1] * 0.1 for o in orientations[0:]]
        zs = [o[:,2] * 0.1 for o in orientations[0:]]

        pos =  [self._data.oMf[self._model.getFrameId(f'{name}_link')].translation for name in self._joint_names]
        for i, (x, p) in enumerate(zip(xs, pos)):
            self._marker_msg.points.append(Point(x=p[0],y=p[1],z=p[2]))
            self._marker_msg.points.append(Point(
                x=float(x[0] + p[0]),
                y=float(x[1] + p[1]),
                z=float(x[2] + p[2])
            ))
            self._marker_msg.colors.append(ColorRGBA(r=1.0, a=1.0, g=(i % 2 / 4) ))# / len(xs) / 2)))
            self._marker_msg.colors.append(ColorRGBA(r=1.0, a=1.0, g=(i % 2 / 4) ))# / len(xs) / 2)))
        for i, (y, p) in enumerate(zip(ys, pos)):
            self._marker_msg.points.append(Point(x=p[0],y=p[1],z=p[2]))
            self._marker_msg.points.append(Point(
                x=float(y[0] + p[0]),
                y=float(y[1] + p[1]),
                z=float(y[2] + p[2])
            ))
            self._marker_msg.colors.append(ColorRGBA(g=1.0, a=1.0, b=(i % 2 / 4) ))# / len(xs) / 2)))
            self._marker_msg.colors.append(ColorRGBA(g=1.0, a=1.0, b=(i % 2 / 4) ))# / len(xs) / 2)))
        for i, (z, p) in enumerate(zip(zs, pos)):
            self._marker_msg.points.append(Point(x=p[0],y=p[1],z=p[2]))
            self._marker_msg.points.append(Point(
                x=float(z[0] + p[0]),
                y=float(z[1] + p[1]),
                z=float(z[2] + p[2])
            ))
            self._marker_msg.colors.append(ColorRGBA(b=1.0, a=1.0, r=(i % 2 / 4) ))# / len(xs) / 2)))
            self._marker_msg.colors.append(ColorRGBA(b=1.0, a=1.0, r=(i % 2 / 4) ))# / len(xs) / 2)))
        self._marker_pub.publish(self._marker_msg)

    def hebi_cb_(self, msg:HebiSensors):
        if self._model is None:
            return
        header = msg.header
        header.frame_id = 'head_link'
        self._marker_msg.header = header
        self._marker_msg.points.clear()
        q = np.array(msg.position, dtype=np.float64)
        v = np.array(msg.velocity, dtype=np.float64)
        pin.forwardKinematics(self._model, self._data, q, v)
        pin.updateFramePlacements(self._model, self._data)

        # Get linear accelerations
        lin_acc = zip(msg.lin_acc.x, msg.lin_acc.y, msg.lin_acc.z)
        lin_acc = self.imu_to_head_frame(lin_acc)

        # Get angular velocities
        ang_vel = zip(msg.ang_vel.x, msg.ang_vel.y, msg.ang_vel.z)
        ang_vel = self.imu_to_head_frame(lin_acc)

        # Predict angular velocities from kinematics
        local_ang_vel = []
        for name in self._joint_names:
            spatial_vel = pin.getFrameVelocity(
                self._model,
                self._data,
                self._model.getFrameId(f'{name}_imu'),
                pin.ReferenceFrame.LOCAL
            )
            local_ang_vel.append(spatial_vel.angular)

        # Get gravity direction
        grav, _, _ = self.get_gravity(lin_acc)

        self._marker_msg.points.append(Point(x=0.0,y=0.0,z=0.0))
        self._marker_msg.points.append(Point(
            x=float(grav[0]),
            y=float(grav[1]),
            z=float(grav[2])
        ))
        self._marker_pub.publish(self._marker_msg)

    def robot_description_cb(self, msg:String):
        self._model = pin.buildModelFromXML(msg.data)
        # self._model = pin.buildModelFromXML(msg.data, pin.JointModelFreeFlyer())
        self._data = self._model.createData()
        self._joint_names = [
            self._model.names[i] for i in range(1, self._model.njoints)
            if "Revolute" in self._model.joints[i].shortname()
        ]

        pin.forwardKinematics(self._model, self._data, np.zeros((self._model.nq,)))
        pin.updateFramePlacements(self._model, self._data)
        self._imu_offsets = []
        for name in self._joint_names:
            frame_id = self._model.getFrameId(f'{name}_link')
            self._imu_offsets.append(deepcopy(self._data.oMf[frame_id]))

    def robot_description_cb_kdl(self, msg:String):
        urdf = Robot.from_xml_string(msg.data)
        for joint in urdf.joints:
            if '_link_to_imu' in joint.name:
                module_name = joint.name.replace('_link_to_imu','')
                self._imu_offsets[module_name] = PyKDL.Frame(
                    PyKDL.Rotation.RPY(*joint.origin.rpy),
                    PyKDL.Vector(*joint.origin.xyz)
                )
        kdl_tree = kdl_tree_from_urdf_model(urdf)
        self._chain = kdl_tree.getChain('head_link', 'tail_link')
        

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()