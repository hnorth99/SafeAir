import serial
import time
import urllib.request
import requests
import codecs
# sudo rfcomm connect /dev/rfcomm1 98:D3:91:FD:91:A6 1
stmDevicesPort = serial.Serial("/dev/rfcomm1", baudrate=9600)
def read_capacity():
    #stmDevicesPort.write(str.encode('1')) #send '1' to Arduino
    #e = stmDevicesPort.readline()
#     if len(list(e)) < 5 and len(list(e)) > 2:
    #stmDevicesPort.reset_input_buffer()
    #stmDevicesPort.reset_output_buffer()
    #print(stmDevicesPort.in_waiting)
    e = stmDevicesPort.readline()
    
    print(e.decode("utf-8"))
    
    #if (len(list(e)) < 5):
    #    print(e.decode("utf-8"))
    
    #serial_read = stmDevicesPort.readline().decode("utf-8") #receive data from Arduino
    #data = (serial_read)
    #return data
    return 44

while True:
    print('start')
    read_capacity()
    print('next')
    time.sleep(10)