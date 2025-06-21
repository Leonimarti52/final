import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Imu
from std_msgs.msg import Header
from cv_bridge import CvBridge
import depthai as dai
import numpy as np
import struct
from scipy.ndimage import median_filter
from sklearn.neighbors import NearestNeighbors
import cv2
import time

class OakdPointCloudAndIMUPublisher(Node):
    def __init__(self):
        super().__init__('oakd_pointcloud_and_imu_publisher')
        
        # Publishers
        self.pointcloud_publisher = self.create_publisher(PointCloud2, 'oakd/points', 10)
        self.imu_publisher = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.bridge = CvBridge()

        # Configure pipeline for depth and color cameras
        self.pipeline = dai.Pipeline()
        self.configure_depth_camera()
        self.configure_color_camera()
        self.configure_imu()

        # Start device
        self.device = dai.Device(self.pipeline)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
        self.q_color = self.device.getOutputQueue("color", maxSize=4, blocking=False)
        self.imu_queue = self.device.getOutputQueue(name="imu", maxSize=50, blocking=False)

        # Create timers
        self.create_timer(1.0/10, self.pointcloud_timer_callback)
        self.create_timer(0.1, self.imu_timer_callback)
        self.get_logger().info("OAK-D PointCloud and IMU Publisher started")

    def configure_depth_camera(self):
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

    def configure_color_camera(self):
        color_cam = self.pipeline.create(dai.node.ColorCamera)
        color_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        color_cam.setPreviewSize(640, 480)
        color_cam.setFps(30)

        xout_color = self.pipeline.create(dai.node.XLinkOut)
        xout_color.setStreamName("color")
        color_cam.preview.link(xout_color.input)

    def configure_imu(self):
        imu = self.pipeline.create(dai.node.IMU)
        xout_imu = self.pipeline.create(dai.node.XLinkOut)
        xout_imu.setStreamName("imu")

        imu.enableFirmwareUpdate(True)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        imu.out.link(xout_imu.input)

        self.get_logger().info("Updating IMU firmware... This may take a few seconds.")
        time.sleep(5)  # Give time to complete the update

    def pointcloud_timer_callback(self):
        in_depth = self.q_depth.tryGet()
        in_color = self.q_color.tryGet()
        if in_depth is None or in_color is None:
            return

        depth_frame = in_depth.getFrame().astype(np.float32)
        color_frame = in_color.getCvFrame()  # color_frame is HxWx3 RGB numpy array

        height, width = depth_frame.shape

        # Median filtering to smooth depth noise
        depth_frame = median_filter(depth_frame, size=3)

        # Fictitious parameters (adjust according to calibration)
        fx = 700.0
        fy = 700.0
        cx = width / 2
        cy = height / 2

        # Downsampling factor for performance
        step = 4

        points = []
        for v in range(0, height, step):
            for u in range(0, width, step):
                z = depth_frame[v, u] / 1000.0  # mm to meters
                if z < 0.1 or z > 10.0:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                # Get corresponding color from the color image
                color_h, color_w, _ = color_frame.shape
                color_u = int(u * (color_w / width))
                color_v = int(v * (color_h / height))

                # Ensure indices are within the color frame
                color_u = np.clip(color_u, 0, color_w - 1)
                color_v = np.clip(color_v, 0, color_h - 1)

                rgb = color_frame[color_v, color_u]  # RGB uint8 array

                # Pack RGB into uint32 0x00RRGGBB
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
                rgb_packed = (r << 16) | (g << 8) | b

                points.append([x, y, z, rgb_packed])

        points = np.array(points)

        # Apply Statistical Outlier Removal filter
        if len(points) > 0:
            points = self.statistical_outlier_removal(points)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "oakd_frame"

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        pc_data = []
        for pt in points:
            x, y, z, rgb = pt
            packed = struct.pack('<fffI', float(x), float(y), float(z), int(rgb))
            pc_data.append(packed)

        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 16
        pointcloud_msg.row_step = pointcloud_msg.point_step * len(points)
        pointcloud_msg.is_dense = False
        pointcloud_msg.data = b''.join(pc_data)

        self.pointcloud_publisher.publish(pointcloud_msg)

    def imu_timer_callback(self):
        imu_data = self.imu_queue.get()
        
        # Create ROS2 IMU message
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        
        # Accelerometer data
        accel = imu_data.packets[0].acceleroMeter
        msg.linear_acceleration.x = accel.x
        msg.linear_acceleration.y = accel.y
        msg.linear_acceleration.z = accel.z
        
        # Gyroscope data (if enabled)
        if len(imu_data.packets) > 1:
            gyro = imu_data.packets[1].gyroscope
            msg.angular_velocity.x = gyro.x
            msg.angular_velocity.y = gyro.y
            msg.angular_velocity.z = gyro.z
        
        self.imu_publisher.publish(msg)

    def statistical_outlier_removal(self, points, nb_neighbors=30, std_ratio=1.0):
        if len(points) == 0:
            return points

        # Use NearestNeighbors to find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=nb_neighbors).fit(points[:, :3])
        distances, _ = nbrs.kneighbors(points[:, :3])

        # Calculate mean distance to neighbors
        mean_distances = np.mean(distances, axis=1)
        std_distances = np.std(distances, axis=1)

        # Filter points that are beyond std_ratio
        mask = (mean_distances < (mean_distances + std_ratio * std_distances))
        return points[mask]

def main(args=None):
    rclpy.init(args=args)
    node = OakdPointCloudAndIMUPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
