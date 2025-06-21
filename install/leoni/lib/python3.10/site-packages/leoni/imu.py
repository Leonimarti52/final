import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import depthai as dai
import time

class DepthAIIMUNode(Node):
    def __init__(self):
        super().__init__('depthai_imu_node')
        self.publisher = self.create_publisher(Imu, 'imu/data_raw', 10)
        
        # Create pipeline
        self.pipeline = dai.Pipeline()
        
        # Configure IMU node
        imu = self.pipeline.create(dai.node.IMU)
        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("imu")
        
        # Enable firmware update (only needed once)
        imu.enableFirmwareUpdate(True)
        
        # Configure IMU sensors (after update)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
        
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        
        # Connect nodes
        imu.out.link(xout.input)
        
        # Connect to device
        self.device = dai.Device(self.pipeline)
        self.get_logger().info("Updating IMU firmware... This may take a few seconds.")
        time.sleep(5)  # Give time to complete the update
        
        # Now we can use the IMU normally
        self.imu_queue = self.device.getOutputQueue(name="imu", maxSize=50, blocking=False)
        
        # Create timer for reading IMU data
        self.timer = self.create_timer(0.1, self.timer_callback)
    
    def timer_callback(self):
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
        
        self.publisher.publish(msg)
        
    def __del__(self):
        if hasattr(self, 'device'):
            self.device.close()

def main(args=None):
    rclpy.init(args=args)
    node = DepthAIIMUNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()