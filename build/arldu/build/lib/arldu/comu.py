import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import serial
import time
import math

class SerialReaderNode(Node):
    def __init__(self):
        super().__init__('serial_reader_node')

        # Configuración de publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.path_pub = self.create_publisher(Path, '/trajectory', 10)
        
        # Broadcasters TF
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Inicialización de Path
        self.path = Path()
        self.path.header.frame_id = 'odom'  # Frame de referencia importante

        # Configuración del puerto serial
        self.serial_setup()

        # Publicar transformación estática world->odom
        self.publish_static_transform()

        # Timer para lectura serial (20Hz)
        self.timer = self.create_timer(0.05, self.read_serial)

    def publish_static_transform(self):
        """Publica la transformación estática world->odom"""
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = 'world'  # Frame fijo
        static_transform.child_frame_id = 'odom'    # Frame móvil
        
        # Inicialmente coinciden (origen en 0,0,0)
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        static_transform.transform.rotation.w = 1.0  # Quaternion identidad
        
        self.static_tf_broadcaster.sendTransform(static_transform)
        self.get_logger().info("Transformación estática world->odom configurada")

    def serial_setup(self):
        """Configura la conexión serial con el Arduino"""
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            time.sleep(2)  # Espera inicialización
            self.get_logger().info("Conexión serial establecida correctamente")
        except Exception as e:
            self.get_logger().error(f"Error al conectar con Arduino: {e}")
            raise

    def read_serial(self):
        """Lee y procesa datos del puerto serial"""
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                self.get_logger().debug(f"Dato recibido: {line}")
                
                # Parseo de datos (x,y,theta)
                parts = line.split(',')
                if len(parts) != 3:
                    self.get_logger().warn(f"Formato incorrecto: {line}")
                    return
                
                x, y, theta = map(float, parts)
                self.process_odometry(x, y, theta)
                
            except ValueError as e:
                self.get_logger().warn(f"Dato inválido: {line} - Error: {e}")
            except Exception as e:
                self.get_logger().error(f"Error inesperado: {e}")

    def process_odometry(self, x, y, theta):
        """Procesa los datos de odometría y publica los mensajes"""
        now = self.get_clock().now().to_msg()
        
        # 1. Crear mensaje Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.orientation = self.yaw_to_quaternion(theta)
        
        # 2. Publicar transformación odom->base_link
        transform = TransformStamped()
        transform.header.stamp = now
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)
        
        # 3. Actualizar trayectoria
        self.update_path(x, y, odom_msg.pose.pose.orientation, now)
        
        # 4. Publicar odometría
        self.odom_pub.publish(odom_msg)
        
        self.get_logger().info(f"Posición publicada - X: {x:.2f}, Y: {y:.2f}, θ: {theta:.2f}")

    def update_path(self, x, y, orientation, timestamp):
        """Actualiza el mensaje Path con la nueva posición"""
        pose = PoseStamped()
        pose.header.stamp = timestamp
        pose.header.frame_id = 'odom'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = orientation
        
        self.path.poses.append(pose)
        
        # Limitar el historial para no consumir mucha memoria
        if len(self.path.poses) > 1000:
            self.path.poses.pop(0)
        
        self.path.header.stamp = timestamp
        self.path_pub.publish(self.path)

    def yaw_to_quaternion(self, theta):
        """Convierte ángulo yaw a quaternion"""
        q = Quaternion()
        q.z = math.sin(theta / 2.0)
        q.w = math.cos(theta / 2.0)
        return q

    def destroy_node(self):
        """Limpieza al cerrar el nodo"""
        self.ser.close()
        self.get_logger().info("Puerto serial cerrado correctamente")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SerialReaderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Nodo detenido por usuario")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()