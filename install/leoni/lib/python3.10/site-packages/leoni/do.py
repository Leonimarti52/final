import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import depthai as dai

class OAKDNode(Node):
    def __init__(self):
        super().__init__('oakd_node')
        self.bridge = CvBridge()

        # Publishers for color, left, and right images
        self.color_publisher = self.create_publisher(Image, 'oakd/color/image_raw', 10)
        self.left_publisher = self.create_publisher(Image, 'oakd/left/image_raw', 10)
        self.right_publisher = self.create_publisher(Image, 'oakd/right/image_raw', 10)

        # Create DepthAI pipeline
        self.pipeline = dai.Pipeline()

        # Color camera setup
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)
        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb_video")
        cam_rgb.preview.link(xout_rgb.input)

        # Left camera setup
        cam_left = self.pipeline.createMonoCamera()
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        xout_left = self.pipeline.createXLinkOut()
        xout_left.setStreamName("left_video")
        cam_left.out.link(xout_left.input)

        # Right camera setup
        cam_right = self.pipeline.createMonoCamera()
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        xout_right = self.pipeline.createXLinkOut()
        xout_right.setStreamName("right_video")
        cam_right.out.link(xout_right.input)

        # Connect to OAK-D device and get output queues
        self.device = dai.Device(self.pipeline)
        self.rgb_queue = self.device.getOutputQueue(name="rgb_video", maxSize=4, blocking=False)
        self.left_queue = self.device.getOutputQueue(name="left_video", maxSize=4, blocking=False)
        self.right_queue = self.device.getOutputQueue(name="right_video", maxSize=4, blocking=False)

        # Create a timer to publish images (30fps)
        self.timer = self.create_timer(1/30, self.timer_callback)

    def timer_callback(self):
        # Process Color Camera
        rgb_frame = None
        video_frame = self.rgb_queue.tryGet()
        if video_frame is not None:
            rgb_frame = video_frame.getCvFrame()
        if rgb_frame is not None:
            msg = self.bridge.cv2_to_imgmsg(rgb_frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.color_publisher.publish(msg)

        # Process Left Camera
        left_frame = None
        left_video_frame = self.left_queue.tryGet()
        if left_video_frame is not None:
            left_frame = left_video_frame.getCvFrame()
        if left_frame is not None:
            # For monochrome cameras, use 'mono8' encoding
            msg = self.bridge.cv2_to_imgmsg(left_frame, encoding="mono8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.left_publisher.publish(msg)

        # Process Right Camera
        right_frame = None
        right_video_frame = self.right_queue.tryGet()
        if right_video_frame is not None:
            right_frame = right_video_frame.getCvFrame()
        if right_frame is not None:
            # For monochrome cameras, use 'mono8' encoding
            msg = self.bridge.cv2_to_imgmsg(right_frame, encoding="mono8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.right_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = OAKDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.device.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()