# Configuring Bluetooth devices
# Open terminal
# Run "hcitool scan"
# Copy address next to HC-06 on ArduinoMega/SCD41
#       98:D3:91:FD:92:50
# Copy address next to HC-06 on ArduinoNano/ResponseDevices
#       98:D3:91:FD:91:A6
# Stop any running scripts
# Run "sudo rfcomm connect /dev/rfcomm0 <SCD41_DEV_ADDR> 1"
# Run "sudo rfcomm connect /dev/rfcomm1 <RESP_DEV_ADDR> 1"

import serial
import time
import urllib.request
import requests


PRINT_FLAG = True

scd41Port = serial.Serial("/dev/rfcomm0", baudrate=9600)
responseDevicesPort = serial.Serial("/dev/rfcomm1", baudrate=9600)

def read_scd41():
    scd41Port.write(str.encode('1')) #send '1' to Arduino
    serial_read = scd41Port.readline().decode("utf-8") #receive data from Arduino

    data = serial_read.split()
    data = [float(x) for x in data]
    
    if (PRINT_FLAG):
        print("Raw serial read: " + serial_read)
        print("Formatted data: ", data)
    return data

def valid_data(data):
    for num in data:
        if (not isinstance(num, float)):
            if (PRINT_FLAG):
                print("Data deemed invalid: " + num)
            return False
    
    if (PRINT_FLAG):
                print("Data deemed valid")
    return True

def climate_response(co2,  humidity, temp):
    response = ""
    response += ("1" if (temp < 11.0) else "0")
    response += ("1" if (humidity < 24) else "0")
    response += ("1" if (humidity > 60) else "0")
    response += ("1" if (co2 > 1500) else "0")
    responseDevicesPort.write(str.encode(response))
   
    if (PRINT_FLAG):
        print("Response Format: heater/humidifier/dehumidifier/fan")
        print("Response: " + response)
        
def push_data_to_thingspeak(timestamp, co2,  humidity, temp):
    url = "https://api.thingspeak.com/update?api_key="
    key = "VZUNJG9HV7CDJAZC"
    header = "&field2={}&field3={}&field4={}".format(temp, humidity, co2)
    final_url = url+key+header
    data = urllib.request.urlopen(final_url)
        

while True:
    data = read_scd41()
    if valid_data(data):
        timestamp = time.localtime()
        climate_response(data[0],  data[1], data[2])
        push_data_to_thingspeak(timestamp, data[0],  data[1], data[2])
    time.sleep(10)
