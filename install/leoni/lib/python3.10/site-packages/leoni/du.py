import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
import depthai as dai
import numpy as np
import struct
from scipy.ndimage import median_filter
from sklearn.neighbors import NearestNeighbors
import cv2

class OAKDCombinedNode(Node):
    def __init__(self):
        super().__init__('oakd_combined_node')
        self.bridge = CvBridge()

        # Publishers for color, left, right images, and point cloud
        self.color_publisher = self.create_publisher(Image, 'oakd/color/image_raw', 10)
        self.left_publisher = self.create_publisher(Image, 'oakd/left/image_raw', 10)
        self.right_publisher = self.create_publisher(Image, 'oakd/right/image_raw', 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, 'oakd/points', 10)

        # Create DepthAI pipeline
        self.pipeline = dai.Pipeline()

        # --- Stereo Depth Camera Setup ---
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        # It's good practice to set resolution here for both stereo and raw output
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) # Set to 400P for consistency with raw output, or higher if preferred and then crop/resize
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) # Set to 400P

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB) # Align depth to RGB camera

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # --- Color Camera Setup (for image_raw and point cloud color) ---
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB) # Ensure correct color order for CvBridge
        cam_rgb.setPreviewSize(640, 480) # Preview size for image_raw and point cloud association
        cam_rgb.setFps(30)

        xout_rgb_preview = self.pipeline.createXLinkOut()
        xout_rgb_preview.setStreamName("rgb_preview")
        cam_rgb.preview.link(xout_rgb_preview.input)

        # --- Reusing Left and Right Mono Camera Streams for raw image publishing ---
        # Instead of creating new MonoCamera nodes, we link from the existing ones.
        xout_left_raw = self.pipeline.createXLinkOut()
        xout_left_raw.setStreamName("left_raw_video")
        mono_left.out.link(xout_left_raw.input) # Link from the 'mono_left' node

        xout_right_raw = self.pipeline.createXLinkOut()
        xout_right_raw.setStreamName("right_raw_video")
        mono_right.out.link(xout_right_raw.input) # Link from the 'mono_right' node


        # Connect to OAK-D device and get output queues
        self.device = dai.Device(self.pipeline)
        self.get_logger().info("Connected to OAK-D device.")

        # Get output queues for all streams
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.q_color_preview = self.device.getOutputQueue(name="rgb_preview", maxSize=4, blocking=False)
        self.q_left_raw = self.device.getOutputQueue(name="left_raw_video", maxSize=4, blocking=False)
        self.q_right_raw = self.device.getOutputQueue(name="right_raw_video", maxSize=4, blocking=False)

        # Get camera calibration for point cloud generation
        calib_data = self.device.readCalibration()
        # Get stereo camera intrinsics for the aligned color camera
        # Note: If depth is aligned to RGB, use RGB intrinsics for point cloud
        self.intrinsics = calib_data.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB, cam_rgb.getPreviewWidth(), cam_rgb.getPreviewHeight()
        )
        self.fx = self.intrinsics[0][0]
        self.fy = self.intrinsics[1][1]
        self.cx = self.intrinsics[0][2]
        self.cy = self.intrinsics[1][2]

        self.get_logger().info(f"Camera Intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

        # Create a timer to publish images and point cloud (adjust frequency as needed)
        self.timer = self.create_timer(1/20, self.timer_callback) # 20 Hz for combined output
        self.get_logger().info("OAK-D Combined Node initialized and publishing.")

    def timer_callback(self):
        current_time = self.get_clock().now().to_msg()

        # --- Process Color Camera (for image_raw) ---
        color_frame_raw = None
        in_color_preview = self.q_color_preview.tryGet()
        if in_color_preview is not None:
            color_frame_raw = in_color_preview.getCvFrame()
            # Publish color image
            msg_color = self.bridge.cv2_to_imgmsg(color_frame_raw, encoding="rgb8")
            msg_color.header.stamp = current_time
            msg_color.header.frame_id = "oakd_frame"
            self.color_publisher.publish(msg_color)

        # --- Process Left Camera (for image_raw) ---
        left_frame_raw = None
        in_left_raw = self.q_left_raw.tryGet()
        if in_left_raw is not None:
            left_frame_raw = in_left_raw.getCvFrame()
            # Publish left image
            msg_left = self.bridge.cv2_to_imgmsg(left_frame_raw, encoding="mono8")
            msg_left.header.stamp = current_time
            msg_left.header.frame_id = "oakd_frame"
            self.left_publisher.publish(msg_left)

        # --- Process Right Camera (for image_raw) ---
        right_frame_raw = None
        in_right_raw = self.q_right_raw.tryGet()
        if in_right_raw is not None:
            right_frame_raw = in_right_raw.getCvFrame()
            # Publish right image
            msg_right = self.bridge.cv2_to_imgmsg(right_frame_raw, encoding="mono8")
            msg_right.header.stamp = current_time
            msg_right.header.frame_id = "oakd_frame"
            self.right_publisher.publish(msg_right)

        # --- Process Depth and generate Point Cloud ---
        in_depth = self.q_depth.tryGet()
        if in_depth is None or color_frame_raw is None: # Need color frame for colored point cloud
            return

        depth_frame = in_depth.getFrame().astype(np.float32) # Depth in millimeters

        height, width = depth_frame.shape

        # Median filter for depth noise reduction
        depth_frame = median_filter(depth_frame, size=3)

        # Downsampling factor for performance (reduce point cloud density)
        step = 2 # Changed to 2 for higher density than original point cloud script (4)

        points = []
        for v in range(0, height, step):
            for u in range(0, width, step):
                z = depth_frame[v, u] / 1000.0  # Convert mm to meters
                if z < 0.1 or z > 10.0:  # Filter out very close/far points
                    continue

                # Reproject depth to 3D point using camera intrinsics
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy

                # Get the corresponding color from the color_frame_raw (which is RGB from preview)
                # Ensure indices are within the color frame boundaries
                color_h, color_w, _ = color_frame_raw.shape
                # Map depth frame coordinates to color frame coordinates (preview size 640x480)
                # Assumes depth frame resolution is the same as the preview size or aligned to it.
                # If depth and color preview have different resolutions,
                # you might need to scale u, v to match color_frame_raw's dimensions.
                # Here, we assume depth_frame.shape is (480, 640) matching previewSize.
                color_u = np.clip(u, 0, color_w - 1)
                color_v = np.clip(v, 0, color_h - 1)

                rgb = color_frame_raw[color_v, color_u] # RGB uint8 array (from cam_rgb.setColorOrder(RGB))

                # Pack RGB into uint32 0x00RRGGBB
                r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
                rgb_packed = (r << 16) | (g << 8) | b

                points.append([x, y, z, rgb_packed])

        points = np.array(points)

        # Apply Statistical Outlier Removal
        if len(points) > 0:
            points = self.statistical_outlier_removal(points)

        # Create and publish PointCloud2 message
        header = Header()
        header.stamp = current_time
        header.frame_id = "oakd_frame" # Consistent frame ID

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        pc_data = []
        for pt in points:
            x, y, z, rgb_packed = pt
            packed = struct.pack('<fffI', float(x), float(y), float(z), int(rgb_packed))
            pc_data.append(packed)

        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 16 # Bytes per point (4 for x, 4 for y, 4 for z, 4 for rgb)
        pointcloud_msg.row_step = pointcloud_msg.point_step * len(points)
        pointcloud_msg.is_dense = False # Can contain NaNs or infs if filters remove points
        pointcloud_msg.data = b''.join(pc_data)

        self.pointcloud_publisher.publish(pointcloud_msg)

    def statistical_outlier_removal(self, points, nb_neighbors=30, std_ratio=1.0):
        if len(points) == 0:
            return points

        # Use NearestNeighbors to find the nearest neighbors
        # Only consider x, y, z for distance calculation
        nbrs = NearestNeighbors(n_neighbors=nb_neighbors).fit(points[:, :3])
        distances, _ = nbrs.kneighbors(points[:, :3])

        # Calculate the mean distance to neighbors
        mean_distances = np.mean(distances, axis=1)

        # Calculate standard deviation of distances
        std_distances = np.std(mean_distances) # Standard deviation of the mean distances

        # Filter points that are beyond std_ratio * standard deviation from the mean of mean distances
        # A common approach is to remove points where mean_distance > mean_of_all_mean_mean_distances + std_ratio * std_dev_of_mean_distances
        mean_of_all_mean_distances = np.mean(mean_distances)
        threshold = mean_of_all_mean_distances + std_ratio * std_distances

        mask = (mean_distances < threshold)
        return points[mask]

def main(args=None):
    rclpy.init(args=args)
    node = OAKDCombinedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'device') and node.device is not None:
            node.device.close()
            node.get_logger().info("OAK-D device closed.")
        node.destroy_node()
        rclpy.shutdown()
        print("ROS 2 shutdown.")

if __name__ == '__main__':
    main()