yolo_face_detector = cv.dnn.readNetFromDarknet('yolov3-face.cfg' , 'yolov3-wider_16000.weights')

def yoloDetectMaskAndDisplay(frame):
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (416, 416),
                                     [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    yolo_face_detector.setInput(blob)
    outs = yolo_face_detector.forward(getOutputsNames(yolo_face_detector))

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
                # Add rest here
            
    #Add rest here


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

