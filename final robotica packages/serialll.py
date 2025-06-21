import serial
import time

# Configura el puerto serie
arduino_port = '/dev/ttyUSB0'  # Cambia según tu sistema
baud_rate = 115200

# Inicializa la conexión serie
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Espera a que se establezca la conexión

output_file = "velocidad.txt"

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            try:
                val1, val2, val3, val4 = map(float, line.split(','))
                print(f"Valor 1: {val1} | Valor 2: {val2} | Valor 3: {val3} | Valor 4: {val4}")

                # Guardar en el archivo (sobrescribe cada vez)
                with open(output_file, 'w') as f:
                    f.write(f"{val1},{val2},{val3},{val4}\n")

            except ValueError:
                print(f"❌ Línea con formato incorrecto: {line}")
except KeyboardInterrupt:
    print("Lectura interrumpida por el usuario.")
finally:
    ser.close()
