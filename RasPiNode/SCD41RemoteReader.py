# Configuring Bluetooth devices
# Open terminal
# Run "hcitool scan"
# Copy address next to HC-06 on ArduinoMega/SCD41
#       98:D3:91:FD:92:50
# Copy address next to HC-06 on ArduinoNano/ResponseDevices
#       TODO
# Stop any running scripts
# Run "sudo rfcomm connect hc10 <SCD41_DEV_ADDR>"
# Run "sudo rfcomm connect hc10 <RESP_DEV_ADDR>"

import serial
import time
import urllib.request
import requests

PRINT_FLAG = True

scd41Port = serial.Serial("/dev/rfcomm0", baudrate=9600)
responseDevicesPort = serial.Serial("/dev/rfcomm1", baudrate=9600)

while True:
    if (PRINT_FLAG):
        print("BEGINNING NEW RUN")
    
    data = read_scd41()
    if valid_data(data):
        timestamp = time.local_time()
        climate_response(data[0],  data[1], data[2])
        push_data_to_thingspeak(timestamp, data[0],  data[1], data[2])
    time.sleep(10)

def read_scd41():
    scd41Port.write(str.encode('1')) #send '1' to Arduino
    serial_read = scd41Port.readline() #receive data from Arduino
    # TODO: FORMAT SERIAL DATA INTO ARRAY
    data = []
    data.append() # TEMP
    data.append() # HUMIDITY
    data.append() # CO2
    
    if (PRINT_FLAG):
        print("Raw serial read: " + serial_read)
        print("Formatted data: " + data)
    return data

def valid_data(data):
    for num in data:
        if (!isinstance(num, float)):
            if (PRINT_FLAG):
                print("Data deemed invalid: " + num)
            return False
    
    if (PRINT_FLAG):
                print("Data deemed valid")
    return True

def climate_response(temp,  humidity, co2):
    response = ""
    response += ((temp < 11.0) ? "1" : "0")
    response += ((humidity < 40) ? "1" : "0")
    response += ((humidity > 60) ? "1" : "0")
    response += ((co2 > 1500) ? "1" : "0")
    responseDevicesPort.write(str.encode(response))
          
    if (PRINT_FLAG):
        print("Response Format: heater/humidifier/dehumidifier/fan")
        print("Response       : " + response)
        
def push_data_to_thingspeak(timestamp, temp,  humidity, co2):
    url = "https://api.thingspeak.com/update?api_key="
    key = "VZUNJG9HV7CDJAZC"
    header = "&Time={}&Temperature={}&Humidity={}&Carbon_Dioxide={}".format(timestamp, temp, humidity, co2)
    final_url = url+key+header
    data = urllib.request.urlopen(final_url)
          
    if (PRINT_FLAG):
        print("Request return: " + data)
