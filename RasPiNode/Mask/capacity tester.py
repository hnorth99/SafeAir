import serial
stmPort = serial.Serial("/dev/rfcomm1", baudrate=9600)

def read_capacity():
    stmPort.write(str.encode('1')) #send '1' to Arduino
    print('here')
    serial_read = stmPort.readline()
    print('here2')
    test = serial_read.decode("utf-8") #receive data from Arduino
    print('here3')
    data = int(serial_read)
    return data

i = 0
while True:
    print(i)
    print(read_capacity())