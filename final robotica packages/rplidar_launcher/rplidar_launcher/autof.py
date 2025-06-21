#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
import math
import os

class MecanumOdomNode(Node):
    def __init__(self):
        super().__init__('mecanum_odom_node')

        # Parámetros del robot
        self.R = 0.03    # Radio de rueda (m)
        self.L = 0.15    # Largo entre ruedas
        self.W = 0.15    # Ancho entre ruedas
        self.dt = 0.05   # Tiempo de muestreo

        # Estado de odometría
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.dist_total = 0.0

        # Publicadores
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer
        self.file_path = "/home/gerkio/ros2_ws/velocidad.txt"
        self.timer = self.create_timer(self.dt, self.read_rpm_file)

    def read_rpm_file(self):
        if not os.path.exists(self.file_path):
            self.get_logger().warn("Archivo velocidad.txt no encontrado")
            return

        try:
            with open(self.file_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    return
                rpm = list(map(float, line.split(',')))
                if len(rpm) != 4:
                    self.get_logger().warn(f"Formato inválido: {line}")
                    return
                self.update_odometry(rpm)
        except Exception as e:
            self.get_logger().error(f"Error leyendo el archivo: {e}")

    def update_odometry(self, rpm):
        # RPM → rad/s
        w = [r * 2 * math.pi / 60.0 for r in rpm]  # 4 ruedas

        # Modelo mecanum directo (vx, vy, w) en base_link
        vx = (self.R / 4) * (w[0] + w[1] + w[2] + w[3])
        vy = -((self.R / 4) * (-w[0] + w[1] + w[2] - w[3]))

        wz = (self.R / (4 * (self.L + self.W))) * (-w[0] + w[1] - w[2] + w[3])

        # Actualización de pose
        delta_x = (vx * math.cos(self.theta) - vy * math.sin(self.theta)) * self.dt
        delta_y = (vx * math.sin(self.theta) + vy * math.cos(self.theta)) * self.dt
        delta_theta = wz * self.dt

        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        self.dist_total += math.sqrt(delta_x**2 + delta_y**2)

        # Imprimir por consola
        print(f"📍 x: {self.x:.3f} m | y: {self.y:.3f} m | θ: {math.degrees(self.theta)%360:.1f}° | 📏 Dist: {self.dist_total:.3f} m")

        self.publish_odometry(vx, vy, wz)

    def publish_odometry(self, vx, vy, wz):
        now = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = self.yaw_to_quaternion(self.theta)

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation = self.yaw_to_quaternion(self.theta)
        self.tf_broadcaster.sendTransform(tf)

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

def main(args=None):
    rclpy.init(args=args)
    node = MecanumOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
