README LIDAR

package ---> rplidar_ros --- libreria para el funcionamiento del lidar
	se ejecuta el lidar de independiente con: 
		ros2 launch rplidar_ros rplidar_a1_launch.py serial_port:=/dev/ttyUSB0 serial_baudrate:=115200			(/dev/ttyUSB0 ----puede variar segun el puerto, puede ser 1,2,3)
		
		
CODIGO DE LECTURA DE VALORES DE ESP32

	serialll.py ----> el txt almacenado esta con (velocidad.txt) este va actualizando su contenido en funcion de la velocidad leida por los encoder mandados por UART (/dev/ttyUSB0 ----puede variar segun el puerto, puede ser 1,2,3)
	
LAUNCHER DE ODOMETRIA CON ENCODERS Y COMPLEMENTOS

package ---> rpilidar_launher
	contenido: autof.py (dentro de rplidar_launcher)---> lee el valor de velocidad.txt y calcula la odometria del vehiculo en funcion de la velocidad de los encoders del auto para ejecutar de forma independiente ros2 run rpilidar_launcher autof
	
	dentro de setup.py y package.xml las respectivas configuraciones para que compile
	start_rplidar.launch.py(dentro de launch)----> configuraciones para la ejecucion del lidar el nodo de odometria y configuraciones previas para map y slam
	codigo de ejecucion del launcher: ros2 launch rplidar_launcher start_rplidar.launch.py
	
	

