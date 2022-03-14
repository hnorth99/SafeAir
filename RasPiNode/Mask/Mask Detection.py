#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

import cv2 as cv
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import board
import adafruit_mlx90614
import time
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
import random
from losantmqtt import Device
#phase 1 imports
import serial
import time
import urllib.request
import requests

# ## Download Face Detectors
cascade_face_detector = cv.CascadeClassifier()
cascade_face_detector.load('haarcascade_frontalface_default.xml')


# OpenCV DNN Face Detection Model downloaded at
# https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel 
dnn_face_detector = cv.dnn.readNet('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

yolo_face_detector = cv.dnn.readNetFromDarknet('yolov3-face.cfg' , 'yolov3-wider_16000.weights')

# Load Mask Detection Models
model = keras.models.load_model('mask_detector')

# Load the TFLite model in TFLite Interpreter
interpreter = tflite.Interpreter('model.tflite')

#allocate the tensors
interpreter.allocate_tensors()

#get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ## Display Detected Face in Window

def cascadeDetectFaceAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    # Detect faces
    faces = face_detector.detectMultiScale(frame_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y+ h), (255, 0, 255), 4)
    frame = cv.resize(frame, (640,640))
    cv.imshow('Capture - Face detection', frame)


def dnnDetectFaceAndDisplay(frame):
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    (h, w) = frame.shape[:2]
    # construct a blob from the image
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    dnn_face_detector.setInput(blob)
    detections = dnn_face_detector.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
    
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.6:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            frame = cv.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 255), 2)

            
    frame = cv.resize(frame, (640,480))
    cv.imshow('Capture - Face detection', frame)


# ## Display Detected Face and Mask Prediction in Window

def dnnDetectMaskAndDisplay(frame):
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    (h, w) = frame.shape[:2]
    # construct a blob from the image
    blob = cv.dnn.blobFromImage(frame, 1.0, (100, 100),
        (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    dnn_face_detector.setInput(blob)
    detections = dnn_face_detector.forward()
    
    faces = []
    locs = []
    preds = []
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
    
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.3:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = (face - np.amin(face)) / (np.amax(face) - np.amin(face))            
            
            input_shape = input_details[0]['shape']
            input_tensor= np.array(np.expand_dims(face,0), dtype=np.float32)

            #set the tensor to point to the input data to be inferred
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.set_tensor(input_index, input_tensor)
            
            #Run the inference
            interpreter.invoke()
            output_details = interpreter.get_output_details()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            (withoutMask, mask) = np.squeeze(output_data)
            
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            # display the label and bounding box rectangle on the output
            # frame
            cv.putText(frame, label, (startX, startY - 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            
    frame = cv.resize(frame, (640, 480))
    cv.putText(frame, '98.6F', (550, 25),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow('Capture - Face detection', frame)


# ## Detect Face and Make Mask Prediction for Display


def dnnDetectMask(frame):
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    (h, w) = frame.shape[:2]
    # construct a blob from the image
    blob = cv.dnn.blobFromImage(frame, 1.0, (100, 100),
        (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    dnn_face_detector.setInput(blob)
    detections = dnn_face_detector.forward()
    
    label = ''
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
    
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.3:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = (face - np.amin(face)) / (np.amax(face) - np.amin(face))            
            
            input_shape = input_details[0]['shape']
            input_tensor= np.array(np.expand_dims(face,0), dtype=np.float32)

            #set the tensor to point to the input data to be inferred
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.set_tensor(input_index, input_tensor)
            
            #Run the inference
            interpreter.invoke()
            output_details = interpreter.get_output_details()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            (withoutMask, mask) = np.squeeze(output_data)
            
            # determine the class label and color we'll use to draw
            # the bounding box and text
            centerX = startX + (endX - startX)/2
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            labelstartX = (centerX - 30) if label == "Mask" else (centerX - 60)
            
            # display the label and bounding box rectangle on the output
            # frame
            cv.putText(frame, label, (int(labelstartX), startY - 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            
    return (cv.resize(frame, (640, 480)), label)
    
def yoloDetectMask(frame):
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (416, 416),
                                     [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    yolo_face_detector.setInput(blob)
    outs = yolo_face_detector.forward(getOutputsNames(yolo_face_detector))

    label = ''
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.6:

                # compute the (x, y)-coordinates of the bounding box for
                # the object
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                startX = int(center_x - width / 2)
                startY = int(center_y - height / 2)
                endX = startX + width
                endY = startY + height


                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                face = cv.resize(face, (224, 224))
                face = (face - np.amin(face)) / (np.amax(face) - np.amin(face))            
            
                input_shape = input_details[0]['shape']
                input_tensor= np.array(np.expand_dims(face,0), dtype=np.float32)

                #set the tensor to point to the input data to be inferred
                input_index = interpreter.get_input_details()[0]["index"]
                interpreter.set_tensor(input_index, input_tensor)
                
                #Run the inference
                interpreter.invoke()
                output_details = interpreter.get_output_details()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                (withoutMask, mask) = np.squeeze(output_data)
                
                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                # display the label and bounding box rectangle on the output
                # frame
                cv.putText(frame, label, (startX, startY - 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    return (cv.resize(frame, (640, 480)), label)


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    

# ## Read Body Temperature

def readTemp(mlx, temp, validTemp):
    global starttime
    measured = mlx.object_temperature
    if (measured >= 33):
        starttime = time.time()
        if ((validTemp == 0) | (measured > temp)):
            temp = measured
        validTemp = 1
    else:
        validTemp = 0
        if (not('starttime' in globals())):
            temp = 0
        elif ((time.time() - starttime) > 10):
            temp = 0
    return temp, validTemp

# Node phase 1 stuff
# Configuring Bluetooth devices
# Open terminal
# Run "hcitool scan"
# Copy address next to HC-06 on ArduinoMega/SCD41
#       98:D3:91:FD:92:50
# Copy address next to HC-06 on STM
#       98:D3:91:FD:91:A6
#       
# Stop any running scripts
# Run "sudo rfcomm connect /dev/rfcomm0 98:D3:91:FD:92:50"
# Run "sudo rfcomm connect /dev/rfcomm1 98:D3:91:FD:91:A6 1"

scd41Port = serial.Serial("/dev/rfcomm0", baudrate=9600)
stmPort = serial.Serial("/dev/rfcomm1", baudrate=9600)
def read_scd41():
    scd41Port.write(str.encode('1')) #send '1' to Arduino
    serial_read = scd41Port.readline().decode("utf-8") #receive data from Arduino

    data = serial_read.split()
    data = [float(x) for x in data]

    return data

def valid_data(data):
    for num in data:
        if (not isinstance(num, float)):
            return False
    return True

def climate_response(co2,  temp, humidity):
    response = ""
    response += ("1" if (temp < 11.0) else "0")   
    response += ("1" if (humidity < 24) else "0")
    response += ("1" if (humidity > 60) else "0")
    response += ("1" if (co2 > 1500) else "0")
    #responseDevicesPort.write(str.encode(response))

def read_capacity():
    stmPort.write(str.encode('1')) #send '1' to Arduino
    serial_read = stmPort.readline().decode("utf-8") #receive data from Arduino
    data = int(serial_read)
    if data < 0:
        data = 0
    return data
    #return 0

'''
# ## Real Time Mask Detection on Raspberry PI

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = frame.array
    
    # show the frame
    dnnDetectMaskAndDisplay(frame)
    key = cv.waitKey(1) & 0xFF
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        
cv.destroyAllWindows()
'''

# ## Real Time Entry Requirement Detection on Raspberry Pi

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

i2c = board.I2C()
mlx = adafruit_mlx90614.MLX90614(i2c)
temp = float(0)
validTemp = 0

# Construct Losant device and connect
device = Device("61fc0cbfc9efb2734ece7825", "56321f25-aa92-4394-8487-9ead9122d2da", 
                "756dc1e9c5f61d52e0f3f3fb662bc241c02bf4f87d48fb4292ebbc6af08510bd")
device.connect(blocking=False)

# Initialize random number generator for capacity
random.seed(34)
cloud_starttime = time.time()

## Phase 3
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
import pytz

location_id = "HunterApt"

firebase_path = "safeairdevice-firebase-adminsdk-ie22z-1ae09cca74.json"
cred = credentials.Certificate(firebase_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://safeairdevice-default-rtdb.firebaseio.com/'
})

def new_location(location):
    firebase_ref = db.reference('/' + location)
    firebase_ref.set({
        "currentcapacity": 0,
        "maxcapacity": 20,
        "capacitycomplaints": 0,
        "co2": None,
        "co2complaints": 0,
        "co2_max": 2200,
        "humidity": None,
        "humiditycomplaints": 0,
        "humidity_max": 80,
        "humidity_min": 40,
        "temperature": None,
        "temperaturecomplaints": 0,
        "temp_min": 22,
        "time": None,
        "password": "abc123",
        "size": 0
        
    })

def read_location(location, metric = ""):
    firebase_ref = db.reference('/' + location + '/' + metric)
    return firebase_ref.get()
    
def climate_update(location, currentcapacity, co2, humidity, temperature):
    firebase_ref = db.reference('/' + location)
    size = read_location(location, "size")
    tz_pst = pytz.timezone('US/Pacific')
    time = datetime.now(tz_pst).strftime("%H:%M:%S")
    
    firebase_ref.update({
        "currentcapacity": currentcapacity,
        "co2/" + str(size): co2,
        "humidity/" + str(size): humidity,
        "temperature/" + str(size): temperature,
        "time/" + str(size): time,
        "size": size + 1
    })
    
co2_max = 1600
humidity_max = 60
humidity_min = 40
temp_min = 11
capacity_max = 100
def get_trigger_params():
    co2_max = read_location(location_id, metric = "co2_max")
    humidity_max = read_location(location_id, metric = "humidity_max")
    humidity_min = read_location(location_id, metric = "humidity_min")
    temp_min = read_location(location_id, metric = "temp_min")
    capacity_max = read_location(location_id, metric = "capacitymax")

def device_trigger(co2, temp, humidity, dev, dev_status):
    get_trigger_params()
    if (dev == 0):
        print(dev_status)
        if (co2 > co2_max and ~(dev_status[dev])):
            dev_status[dev] = True
            print(dev_status)
            requests.get('https://api.voicemonkey.io/trigger?access_token=592c87dcef9a4afc2a043753a9223d53&secret_token=446a94d5cabb17337e49e686da8e7be4&monkey=turn-on-fan')
            print("Turning on fan")
        elif (co2 < co2_max and dev_status[dev]):
            dev_status[dev] = False
            requests.get('https://api.voicemonkey.io/trigger?access_token=592c87dcef9a4afc2a043753a9223d53&secret_token=446a94d5cabb17337e49e686da8e7be4&monkey=turn-off-fan')
       
    if dev == 1:
        print(dev_status)
        if (humidity > humidity_max and ~(dev_status[dev])):
            dev_status[dev] = True
            requests.get('https://api.voicemonkey.io/trigger?access_token=592c87dcef9a4afc2a043753a9223d53&secret_token=446a94d5cabb17337e49e686da8e7be4&monkey=turn-on-dehumidifier')
        elif (humidity < humidity_max and dev_status[dev]):
            requests.get('https://api.voicemonkey.io/trigger?access_token=592c87dcef9a4afc2a043753a9223d53&secret_token=446a94d5cabb17337e49e686da8e7be4&monkey=turn-off-dehumidifier')
            dev_status[dev] = False
    
    if dev == 2:
        if (humidity < humidity_min and ~(dev_status[dev])):
            dev_status[dev] = True
            requests.get('https://api.voicemonkey.io/trigger?access_token=592c87dcef9a4afc2a043753a9223d53&secret_token=446a94d5cabb17337e49e686da8e7be4&monkey=turn-on-humidifier')
        elif (humidity > humidity_min and dev_status[dev]):
            dev_status[dev] = False
            requests.get('https://api.voicemonkey.io/trigger?access_token=592c87dcef9a4afc2a043753a9223d53&secret_token=446a94d5cabb17337e49e686da8e7be4&monkey=turn-off-humidifier')
    if dev == 3:
        if (temp < temp_min and ~(dev_status[dev])):
            dev_status[dev] = True
            requests.get('https://api.voicemonkey.io/trigger?access_token=592c87dcef9a4afc2a043753a9223d53&secret_token=446a94d5cabb17337e49e686da8e7be4&monkey=turn-on-heat')
        elif (temp > temp_min and dev_status[dev]):
            dev_status[dev] = False
            requests.get('https://api.voicemonkey.io/trigger?access_token=592c87dcef9a4afc2a043753a9223d53&secret_token=446a94d5cabb17337e49e686da8e7be4&monkey=turn-off-heat')
    return dev_status
    
dev = 0
dev_status = [False, False, False, False]
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = frame.array
    
    # show the frame
    frame, label = dnnDetectMask(frame)
    temp, validTemp = readTemp(mlx, temp, validTemp)
    adjusted = temp + 2.5
    
    if (label == 'No Mask'):
        cv.rectangle(frame, (140, 43), (527, 95), (0, 0, 255), -1)
        cv.putText(frame, 'ENTRY DENIED', (145, 87),
                cv.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
    elif (label == 'Mask'):
        if ((adjusted <= 38) & (temp != 0)):
            cv.rectangle(frame, (123,43), (545, 95), (0, 255, 0), -1)
            cv.putText(frame, 'ENTRY GRANTED', (120, 87),
                cv.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
        else:
            cv.rectangle(frame, (140, 43), (527, 95), (0, 0, 255), -1)
            cv.putText(frame, 'ENTRY DENIED', (145, 87),
                cv.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 3)
    
    farenheit = (adjusted * 9 / 5) + 32
    
    if (adjusted <= 38):
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    
    if (temp != 0):    
        temp_str = "{:.1f}".format(farenheit) + 'F'
        if (farenheit < 100):
            cv.putText(frame, temp_str, (550, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, color, 2) # placed for 3 digits
        else:
            cv.putText(frame, temp_str, (530, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1, color, 2) # placed for 3 digits
    
    cv.imshow('ENTRY REQUIREMENT CHECKING', frame)
    key = cv.waitKey(1) & 0xFF
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    if ((time.time() - cloud_starttime) > 10):
        # Read and report climate data every 10 seconds   
        device.loop()
        data = read_scd41()
        print(data)
        if valid_data(data):
            capacity = read_capacity() #(read_capacity() /capacity_max)
            print(capacity)
            air_temp = data[1]
            humidity = data[2]
            co2 = data[0]
            timestamp = time.localtime()
            dev_status = device_trigger(co2, air_temp, humidity, dev, dev_status)
            dev = (dev + 1) % 4
            climate_update(location_id, capacity, co2, humidity, air_temp)
            device.send_state({"Temperature": air_temp, "Relative_Humidity": humidity, "Carbon_Dioxide": co2, "Capacity": (capacity /capacity_max)})
            cloud_starttime = time.time()
            
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        
cv.destroyAllWindows()

'''
******************************************************************************
**********************CITATIONS***********************************************
******************************************************************************

MLX90614 Infrared Sensor Python library
https://docs.circuitpython.org/projects/mlx90614/en/latest/

STM32F4 HAL User Manual
file:///C:/Users/melin/AppData/Local/Temp/dm00105879-description-of-stm32f4-hal-and-ll-drivers-stmicroelectronics.pdf

Tutorial and Mask Detection Code: COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning
https://pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/ 

OpenCV Guide to Deep Learning
https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/

OpenCV Library and Documentation
https://docs.opencv.org/ 

Tensorflow API
https://www.tensorflow.org/guide 

Keras API
https://faroit.com/keras-docs/1.2.0/ 

OpenCV DNN Face Detection Model
https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel 

URL Request Library
https://docs.python.org/3/library/urllib.request.html 

Losant MQTT Python Client
https://docs.losant.com/mqtt/python/ 
'''
