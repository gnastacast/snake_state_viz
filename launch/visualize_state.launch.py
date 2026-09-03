from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.actions import IncludeLaunchDescription, SetLaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def check_for_package_path(context, *args, **kwargs):
    # 1. Extract the raw string path
    raw_path = LaunchConfiguration(kwargs.get('arg_name')).perform(context)
    if not raw_path:
        return ""

    # 2. Check if file exists, otherwise prepend the package path
    absolute_path = raw_path
    if not Path(raw_path).exists():
        package_path = Path(kwargs.get('pkg_path'))
        package_base = package_path.parts[0]
        package_abs = Path(get_package_share_directory(str(package_base)))
        package_rel = package_path.relative_to(package_base)
        new_path = package_abs / package_rel / raw_path
        if new_path.exists():
            absolute_path = str(new_path)
        else:
            raise ValueError(f"Neither {absolute_path} nor {new_path} exist")

    # 3. Update the LaunchConfiguration with the absolute path
    return [SetLaunchConfiguration(kwargs.get('arg_name'), absolute_path)]


def generate_launch_description():
    snake_yaml = LaunchConfiguration('snake_yaml')
    sim_yaml = LaunchConfiguration('sim_yaml')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'sim_yaml',
            default_value="sim_params.yaml",
            description='File defining simulation parameters. Leave blank when using a real snake robot. Path can be absolute or relative to "snakelib_bullet/param/',
        ),
        DeclareLaunchArgument(
            'snake_yaml',
            default_value='ruby.yaml',
            description='File defining type of snake and list of modules. Path can be absolute or relative to "snakelib_description/snakes/"'
        ),
        # Intercept and modify snake_yaml path
        OpaqueFunction(
            function=check_for_package_path,
            kwargs={
                'pkg_path': 'snakelib_description/snakes/',
                'arg_name': 'snake_yaml'
            }
        ),
        # Intercept and modify sim_yaml path
        OpaqueFunction(
            function=check_for_package_path,
            kwargs={
                'pkg_path': 'snakelib_bullet/param/',
                'arg_name': 'sim_yaml'
            }
        ),
        # Bringup simulated or real robot
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                Path(
                    get_package_share_directory('snakelib_control'),
                    'launch',
                    'snake_bringup.launch.py'
                )
            ]),
            launch_arguments={
                'sim_yaml': sim_yaml,
                'snake_yaml': snake_yaml
            }.items()
        ),
    ])
