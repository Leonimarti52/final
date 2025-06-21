import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import struct
from random import gauss

class KalmanFilter:
    def __init__(self, initial_state=0.0, initial_uncertainty=1.0, motion_variance=1.0, measurement_variance=1.0):
        self.state_estimate = initial_state
        self.uncertainty = initial_uncertainty
        self.motion_variance = motion_variance
        self.measurement_variance = measurement_variance

    def predict(self, motion):
        self.state_estimate += motion
        self.uncertainty += self.motion_variance

    def update(self, measurement):
        kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_variance)
        self.state_estimate += kalman_gain * (measurement - self.state_estimate)
        self.uncertainty = (1 - kalman_gain) * self.uncertainty

    def get_estimate(self):
        return self.state_estimate

class ParticleFilter:
    def __init__(self, num_particles=1000, init_range=(0, 10)):
        self.num_particles = num_particles
        self.particles = np.random.uniform(init_range[0], init_range[1], num_particles)
        self.weights = np.ones(num_particles) / num_particles

    def motion_update(self, movement, motion_noise=0.5):
        self.particles += movement + np.random.normal(0, motion_noise, self.num_particles)

    def measurement_update(self, measurement, measurement_noise=1.0):
        errors = self.particles - measurement
        self.weights = np.exp(-0.5 * (errors / measurement_noise) ** 2)
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimate(self):
        return np.average(self.particles, weights=self.weights)

class FusionEstimator(Node):
    def __init__(self):
        super().__init__('fusion_estimator')
        self.publisher_ = self.create_publisher(Float32, 'fusion/estimate', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'fusion/points', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.true_position = 0.0
        self.motion_per_step = 1.0

        self.kalman = KalmanFilter()
        self.particle = ParticleFilter()

        self.get_logger().info('Filtro combinado Kalman + Partículas iniciado')

    def timer_callback(self):
        self.true_position += self.motion_per_step
        measurement = self.true_position + gauss(0, 1.0)

        self.kalman.predict(self.motion_per_step)
        self.kalman.update(measurement)
        kalman_est = self.kalman.get_estimate()

        self.particle.motion_update(self.motion_per_step)
        self.particle.measurement_update(measurement)
        self.particle.resample()
        particle_est = self.particle.get_estimate()

        promedio = (kalman_est + particle_est) / 2.0

        # Publicar promedio en Float32
        msg = Float32()
        msg.data = float(promedio)
        self.publisher_.publish(msg)

        # Crear PointCloud2 a partir de partículas (x=valor, y=z=0)
        points = [[p, 0.0, 0.0, 0xFFFFFF] for p in self.particle.particles]  # blanco
        pc_data = b''.join([struct.pack('<fffI', *pt) for pt in points])

        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = Header()
        pointcloud_msg.header.stamp = self.get_clock().now().to_msg()
        pointcloud_msg.header.frame_id = "map"
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)
        pointcloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 16
        pointcloud_msg.row_step = 16 * len(points)
        pointcloud_msg.is_dense = True
        pointcloud_msg.data = pc_data

        self.pointcloud_pub.publish(pointcloud_msg)

        self.get_logger().info(f'True: {self.true_position:.2f} | Kalman: {kalman_est:.2f} | Partícula: {particle_est:.2f} | Prom: {promedio:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = FusionEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
