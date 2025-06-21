import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import depthai as dai

class OAKDNode(Node):
    def __init__(self):
        super().__init__('oakd_node')
        self.bridge = CvBridge()

        # Publicador de imágenes
        self.publisher = self.create_publisher(Image, 'oakd/color/image_raw', 10)

        # Crear pipeline DepthAI
        self.pipeline = dai.Pipeline()

        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)

        xout = self.pipeline.createXLinkOut()
        xout.setStreamName("video")
        cam_rgb.preview.link(xout.input)

        self.device = dai.Device(self.pipeline)
        self.video_queue = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)

        # Crear un timer para publicar imágenes (30fps)
        self.timer = self.create_timer(1/30, self.timer_callback)

    def timer_callback(self):
        frame = None
        video_frame = self.video_queue.tryGet()
        if video_frame is not None:
            frame = video_frame.getCvFrame()
        
        if frame is not None:
            # Convertir frame a mensaje ROS
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = OAKDNode()
    rclpy.spin(node)
    node.device.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
