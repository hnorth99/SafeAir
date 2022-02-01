# Configuring Bluetooth device
# Open terminal
# Run "hcitool scan"
# Copy address next to HC-06
# Stop any running scripts
# Run "sudo rfcomm connect hc10 98:D3:91:FD:92:50"

import serial
import time
scd41Port = serial.Serial("/dev/rfcomm0",baudrate=9600)

while True:
    scd41Port.write(str.encode('1')) #send '1' to Arduino
    data=scd41Port.readline() #receive data from Arduino
    if data:
        print(data.decode('utf-8'))
    time.sleep(0.1)

