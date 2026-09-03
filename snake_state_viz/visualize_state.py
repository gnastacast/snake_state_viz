import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Header
# from snakelib_control.pykdl_utils.kdl_kinematics import KDLKinematics
from snakelib_control.pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from snakelib_msgs.msg import HebiSensors

from urdf_parser_py.urdf import Robot

import PyKDL

from tf2_ros import TransformBroadcaster, TransformStamped


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

def get_fk_all_frames(chain:PyKDL.Chain, joint_angles:list[float]) -> list[PyKDL.Frame]:
    """
    Computes forward kinematics for every segment/frame in a PyKDL.Chain.
    
    :param chain: PyKDL.Chain object containing the kinematic segments.
    :param joint_angles: PyKDL.JntArray containing positions for all movable joints.
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

        print(joint.getName(), joint.getTypeName())
        
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

class MinimalPublisher(Node):
    _chain: PyKDL.Chain | None = None
    _joint_map: dict[int, str] = {}

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

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

    def hebi_cb(self, msg:HebiSensors):
        if self._chain is None:
            return
        if not self._joint_map:
            for i in range(self._chain.getNrOfSegments()):
                name = self._chain.getSegment(i).getName()
                if name in msg.name:
                    self._joint_map[name] = i+1
        header = msg.header
        header.frame_id = 'head_link'

        frames = get_fk_all_frames(self._chain, msg.position)
        for name, frame in zip(msg.name, frames):
            self.tf_broadcaster.sendTransform(frame_to_tf(frame, f'/test/{name}', header))

    def robot_description_cb(self, msg:String):
        urdf = Robot.from_xml_string(msg.data)
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