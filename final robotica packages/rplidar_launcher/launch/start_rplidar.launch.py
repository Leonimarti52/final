from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessStart

def generate_launch_description():
    # Crear el nodo slam_toolbox como LifecycleNode
    slam_node = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='',  # <--- ¡necesario!
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'scan_topic': '/scan',
            'map_update_interval': 0.5,
            'max_laser_range': 5.0,
            'minimum_time_interval': 0.2,
            'transform_publish_period': 0.05,
            'use_pose_extrapolator': True,
            'interpolate_odometry': True,
            'map_update_frequency': 2.0
        }]
    )

    # Manejar configuración y activación del slam_toolbox automáticamente
    lifecycle_events = RegisterEventHandler(
        OnProcessStart(
            target_action=slam_node,
            on_start=[
                TimerAction(
                    period=2.0,
                    actions=[
                        ExecuteProcess(
                            cmd=['ros2', 'lifecycle', 'set', '/slam_toolbox', 'configure'],
                            output='screen'
                        )
                    ]
                ),
                TimerAction(
                    period=4.0,
                    actions=[
                        ExecuteProcess(
                            cmd=['ros2', 'lifecycle', 'set', '/slam_toolbox', 'activate'],
                            output='screen'
                        )
                    ]
                )
            ]
        )
    )

    return LaunchDescription([
        # Parámetros del RPLIDAR
        DeclareLaunchArgument(
            'serial_port',
            default_value='/dev/ttyUSB1',
            description='Puerto serial del RPLIDAR'
        ),
        DeclareLaunchArgument(
            'serial_baudrate',
            default_value='115200',
            description='Baudios del RPLIDAR'
        ),

        # Nodo RPLIDAR
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_node',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'serial_baudrate': LaunchConfiguration('serial_baudrate'),
            }],
            output='screen'
        ),

        # Nodo de odometría
        Node(
            package='rplidar_launcher',
            executable='autof',
            name='autof',
            output='screen'
        ),

        # Nodo de SLAM
        slam_node,
        lifecycle_events,

        # TF: map → odom
        ExecuteProcess(
            cmd=['ros2', 'run', 'tf2_ros', 'static_transform_publisher',
                 '0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),

        # TF: base_link → laser_frame
        ExecuteProcess(
            cmd=['ros2', 'run', 'tf2_ros', 'static_transform_publisher',
                 '0', '0', '0', '0', '0', '0', 'base_link', 'laser_frame'],
            output='screen'
        ),
    ])
