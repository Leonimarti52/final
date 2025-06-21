import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import TransformStamped, PoseWithCovariance, TwistWithCovariance
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import cv2
import time
from collections import deque
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class VIWOdometryNode(Node):
    def __init__(self):
        super().__init__('viw_odometry_node')
        
        # QoS profiles for different sensors
        imu_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        wheel_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, 
            'imu/data_raw', 
            self.imu_callback, 
            imu_qos)
            
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, 
            'oakd/points', 
            self.pointcloud_callback, 
            10)
            
        self.wheel_odom_sub = self.create_subscription(
            Odometry,
            '/odom',  # From your wheel odometry node
            self.wheel_odom_callback,
            wheel_qos)
        
        # Publisher
        self.odom_pub = self.create_publisher(Odometry, 'viw_odometry/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Variables
        self.current_position = np.zeros(3)
        self.current_orientation = R.identity()
        self.current_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
        # Sensor data buffers
        self.imu_buffer = deque(maxlen=100)
        self.wheel_odom_buffer = deque(maxlen=100)
        self.last_pointcloud = None
        
        # Time tracking
        self.last_imu_time = None
        self.last_wheel_time = None
        self.last_pointcloud_time = None
        self.last_update_time = time.time()
        
        # Parameters (tune these based on your system)
        self.accel_noise = 0.1
        self.gyro_noise = 0.01
        self.wheel_noise = 0.05
        self.pointcloud_downsample = 0.1  # downsample ratio
        
        # Sensor weights (0-1)
        self.imu_weight = 0.4
        self.visual_weight = 0.4
        self.wheel_weight = 0.2
        
        # Wheel odometry state
        self.last_wheel_position = np.zeros(3)
        self.wheel_position = np.zeros(3)
        self.wheel_orientation = R.identity()
        
        # Kalman filter variables (simplified)
        self.state_covariance = np.eye(6)  # Position and orientation covariance
        
        self.get_logger().info("Visual-Inertial-Wheel Odometry Node Initialized")

    def imu_callback(self, msg):
        # Store IMU data with timestamp
        current_time = time.time()
        imu_data = {
            'time': current_time,
            'linear_accel': np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ]),
            'angular_vel': np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
        }
        self.imu_buffer.append(imu_data)
        
        # Initial IMU integration if this is the first reading
        if self.last_imu_time is None:
            self.last_imu_time = current_time
            return
            
        # Time since last IMU update
        dt = current_time - self.last_imu_time
        
        # Simple IMU integration (dead reckoning)
        if len(self.imu_buffer) >= 2:
            prev_imu = self.imu_buffer[-2]
            current_imu = self.imu_buffer[-1]
            
            # Average angular velocity over the interval
            avg_angular_vel = (prev_imu['angular_vel'] + current_imu['angular_vel']) / 2
            
            # Update orientation using gyro data
            rotation_vector = avg_angular_vel * dt
            rotation = R.from_rotvec(rotation_vector)
            self.current_orientation = rotation * self.current_orientation
            
            # Average acceleration over the interval
            avg_accel = (prev_imu['linear_accel'] + current_imu['linear_accel']) / 2
            
            # Update position using acceleration (in world frame)
            world_accel = self.current_orientation.apply(avg_accel) - np.array([0, 0, 9.81])  # subtract gravity
            self.current_velocity += world_accel * dt
            self.current_position += self.current_velocity * dt + 0.5 * world_accel * dt**2
        
        self.last_imu_time = current_time
        self.angular_velocity = imu_data['angular_vel']
        
        # Periodically update odometry even if we don't have new visual or wheel data
        if current_time - self.last_update_time > 0.1:  # 10Hz update
            self.publish_odometry()
            self.last_update_time = current_time

    def wheel_odom_callback(self, msg):
        # Store wheel odometry data with timestamp
        current_time = time.time()
        wheel_data = {
            'time': current_time,
            'position': np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]),
            'orientation': R.from_quat([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]),
            'linear_velocity': np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ]),
            'angular_velocity': np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ])
        }
        self.wheel_odom_buffer.append(wheel_data)
        
        # Update wheel odometry state
        self.last_wheel_position = self.wheel_position
        self.wheel_position = wheel_data['position']
        self.wheel_orientation = wheel_data['orientation']
        
        if self.last_wheel_time is None:
            self.last_wheel_time = current_time
            return
            
        # Fuse wheel odometry with current estimate
        self.fuse_sensor_data()
        self.last_wheel_time = current_time

    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to numpy array
        points = self.pointcloud2_to_array(msg)
        
        if points is None or len(points) == 0:
            return
            
        # Downsample point cloud
        if self.pointcloud_downsample < 1.0:
            idx = np.random.choice(len(points), int(len(points)*self.pointcloud_downsample), replace=False)
            points = points[idx]
        
        current_time = time.time()
        
        # First pointcloud - just store it
        if self.last_pointcloud is None:
            self.last_pointcloud = points
            self.last_pointcloud_time = current_time
            return
            
        # Time since last pointcloud
        dt = current_time - self.last_pointcloud_time
        
        # Estimate motion using ICP (Iterative Closest Point)
        try:
            # Convert to Open3D point clouds
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(points[:, :3])
            
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(self.last_pointcloud[:, :3])
            
            # Run ICP with initial guess from IMU
            threshold = 0.1  # max correspondence distance
            
            # Create initial guess from IMU data
            trans_init = np.eye(4)
            if len(self.imu_buffer) > 0:
                # Estimate rotation from IMU
                imu_data = self.imu_buffer[-1]
                rotation_vector = imu_data['angular_vel'] * dt
                rotation = R.from_rotvec(rotation_vector)
                trans_init[:3, :3] = rotation.as_matrix()
                
                # Estimate translation from IMU
                world_accel = self.current_orientation.apply(imu_data['linear_accel']) - np.array([0, 0, 9.81])
                translation = self.current_velocity * dt + 0.5 * world_accel * dt**2
                trans_init[:3, 3] = translation
            
            # Use point-to-plane ICP for better accuracy
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            
            transformation = reg_p2p.transformation
            
            # Extract translation and rotation
            translation = transformation[:3, 3]
            rotation_matrix = transformation[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            
            # Fuse visual odometry with other sensors
            self.fuse_visual_odometry(translation, rotation, current_time)
            
        except Exception as e:
            self.get_logger().warn(f"ICP failed: {str(e)}")
        
        # Store current pointcloud for next iteration
        self.last_pointcloud = points
        self.last_pointcloud_time = current_time
        
        # Publish odometry
        self.publish_odometry()

    def fuse_visual_odometry(self, translation, rotation, current_time):
        # Simple weighted average fusion
        visual_position = self.current_position + self.current_orientation.apply(translation)
        visual_orientation = rotation * self.current_orientation
        
        # Get wheel odometry data if available
        wheel_position = None
        wheel_orientation = None
        if len(self.wheel_odom_buffer) > 0:
            wheel_data = self.wheel_odom_buffer[-1]
            wheel_position = wheel_data['position']
            wheel_orientation = wheel_data['orientation']
        
        # Fuse estimates
        if wheel_position is not None:
            # Weighted average of visual and wheel odometry
            self.current_position = (self.visual_weight * visual_position + 
                                   self.wheel_weight * wheel_position) / (self.visual_weight + self.wheel_weight)
            
            # SLERP for orientation interpolation
            self.current_orientation = R.from_matrix(
                (self.visual_weight * visual_orientation.as_matrix() + 
                 self.wheel_weight * wheel_orientation.as_matrix()) / 
                (self.visual_weight + self.wheel_weight)
            ).as_quat()
        else:
            # Just use visual estimate if no wheel data
            self.current_position = visual_position
            self.current_orientation = visual_orientation
        
        # Update velocity estimate
        if self.last_pointcloud_time is not None and current_time - self.last_pointcloud_time > 0:
            self.current_velocity = translation / (current_time - self.last_pointcloud_time)

    def fuse_sensor_data(self):
        # Simple sensor fusion between IMU, visual, and wheel odometry
        if len(self.imu_buffer) == 0 or len(self.wheel_odom_buffer) == 0:
            return
            
        current_time = time.time()
        
        # Get latest IMU data
        imu_data = self.imu_buffer[-1]
        
        # Get latest wheel odometry data
        wheel_data = self.wheel_odom_buffer[-1]
        
        # Calculate time differences
        imu_dt = current_time - self.last_imu_time if self.last_imu_time else 0
        wheel_dt = current_time - self.last_wheel_time if self.last_wheel_time else 0
        
        # Calculate weights based on time since last update
        total_weight = self.imu_weight + self.visual_weight + self.wheel_weight
        imu_rel_weight = self.imu_weight / total_weight
        wheel_rel_weight = self.wheel_weight / total_weight
        
        # Fuse position estimates
        if wheel_dt < 1.0:  # Only use wheel if recent
            # Weighted average based on configured weights
            self.current_position = (
                imu_rel_weight * self.current_position +
                wheel_rel_weight * wheel_data['position']
            )
            
            # Fuse orientation using SLERP
            imu_orientation = self.current_orientation
            wheel_orientation = wheel_data['orientation']
            
            # Simple weighted average of rotation matrices
            fused_rotmat = (
                imu_rel_weight * imu_orientation.as_matrix() +
                wheel_rel_weight * wheel_orientation.as_matrix()
            )
            
            # Normalize to get valid rotation matrix
            u, s, vh = np.linalg.svd(fused_rotmat)
            self.current_orientation = R.from_matrix(u @ vh)
            
            # Fuse velocities
            self.current_velocity = (
                imu_rel_weight * self.current_velocity +
                wheel_rel_weight * wheel_data['linear_velocity']
            )
            self.angular_velocity = (
                imu_rel_weight * self.angular_velocity +
                wheel_rel_weight * wheel_data['angular_velocity']
            )

    def pointcloud2_to_array(self, cloud_msg):
        # Convert PointCloud2 message to numpy array
        points = np.frombuffer(cloud_msg.data, dtype=np.float32)
        points = points.reshape(-1, 4)  # assuming x,y,z,rgb format
        
        # Remove NaN/inf values
        mask = np.isfinite(points).all(axis=1)
        return points[mask]

    def publish_odometry(self):
        # Create Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Set position
        odom_msg.pose.pose.position.x = self.current_position[0]
        odom_msg.pose.pose.position.y = self.current_position[1]
        odom_msg.pose.pose.position.z = self.current_position[2]
        
        # Set orientation (convert from scipy Rotation to quaternion)
        quat = self.current_orientation.as_quat()
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        
        # Set velocity
        odom_msg.twist.twist.linear.x = self.current_velocity[0]
        odom_msg.twist.twist.linear.y = self.current_velocity[1]
        odom_msg.twist.twist.linear.z = self.current_velocity[2]
        odom_msg.twist.twist.angular.x = self.angular_velocity[0]
        odom_msg.twist.twist.angular.y = self.angular_velocity[1]
        odom_msg.twist.twist.angular.z = self.angular_velocity[2]
        
        # Publish
        self.odom_pub.publish(odom_msg)
        
        # Publish TF
        transform = TransformStamped()
        transform.header.stamp = odom_msg.header.stamp
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = self.current_position[0]
        transform.transform.translation.y = self.current_position[1]
        transform.transform.translation.z = self.current_position[2]
        transform.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    node = VIWOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()