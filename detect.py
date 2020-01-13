#tweak frames and padding of boxes -Global var
frames = 20 # lower value gives more number of frames
padding = 250 # amount of cropping that is specifed from center
x_y_values = []


#Library Imports
import numpy as np
import argparse
import time
import cv2
import os
import tensorflow as tf
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy
#Loading Pre-trained model with specified path as arguments
model = tf.keras.models.load_model('model')

CATEGORIES = ["No", "Yes"]

def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    x = model.predict(new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1))
    if x == 1:
        plt.imshow(new_array, cmap='gray')
        plt.show()
        cv2.imwrite("test/ops/opi.jpg",new_array)
        #service.send_message("9945549200", "Accident Detected at "+location)
        
# print(x)
# return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# a = numpy.array2string(prediction)
# print(a)

def obj(name):
    value = name
    #value = "object_detection/ss{}.jpg".format(j)
    
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(['yolo-coco', "yolov3.weights"])
    configPath = os.path.sep.join(['yolo-coco', "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(value)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
        0.3)
   
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            temp = []
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            

            
#             text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            text = "{}".format(LABELS[classIDs[i]])
#             print(text)
            if text == "car" or text=="truck":
                temp.append([x,y+h])
                temp.append([x+w,y+h])
                temp.append([x+w,y])
                temp.append([x,y])
#                 temp.append([i])
                x_y_values.append(temp)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                cv2.putText(image, str(i)+", "+str(x)+" "+(str(y)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

    # show the output image
    #cv2.imshow("Image", image)
    #cv2.imwrite("test/ops/op.jpg",image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
def crops(value):
    for i in range(len(x_y_values)):
        for j in range (i+1,len(x_y_values)-1):
            box1 = x_y_values[i]
            box2 = x_y_values[j]
            overlap(box1,box2,value)
    
def overlap(box1, box2,value):
    poly_1 = Polygon(box1)
    poly_2 = Polygon(box2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
#     print("overlap", iou*100)
    if iou > 0.02:
        crop(box1,box2, value)
    
def crop(box1, box2, value):

    b1x = box1[1][0] + box1[3][0]
    b1x = b1x/2
    
    b1y = box1[1][1] + box1[3][1]
    b1y = b1y/2
    
    b2x = box2[1][0] + box2[3][0]
    b2x = b2x/2
    
    b2y = box2[1][1] + box2[3][1]
    b2y = b2y/2
    
    c1 = (b1x+b2x)/2 
    c2 = (b1y+b2y)/2 
    img = cv2.imread(value)
    a = c2-padding
    b = c2+padding
    c = c1-padding
    d = c1+padding

    if a < 0:
        a=0
    if b < 0:
        b=0
    if c < 0:
        c=0
    if d < 0:
        d=0
#     print(a)
#     print(b)
#     print(c)
#     print(d)
    crop_img = img[int(a):int(b),int(c):int(d)]
    cv2.imwrite( "test/crops/cropped.jpg",crop_img)
    prediction = prepare('test/crops/cropped.jpg')

def rund(path):


    if path == 'demo.mp4':
        padding = 250
        location = 'loc1'
    elif path == 'demo4.mp4':
        padding = 75
        location = 'loc2'
    else:
        padding = 250
        location = 'loc3'

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(path)
    j = 0
    i=0

    while True:

        
        ret, frame = cap.read()
        
        if ret: # Change Made 
        
            #cv2.imshow('frame', frame)
            #print(frame)
            frame = cv2.resize(frame,(400,300))
            if i%20 == 0:
                j = j+1
                s = "test/ss{}.jpg".format(j)
                cv2.imwrite(s,frame)
                obj(s)
                crops(s)
                x_y_values = []
            i=i+1



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
            
    cap.release()
    cv2.destroyAllWindows()
