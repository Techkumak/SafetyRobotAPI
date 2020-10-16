from django.shortcuts import render
from django.http import Http404
from rest_framework.decorators import api_view
import json
from django.http import JsonResponse 
from cv2 import cv2
import numpy as np
import base64
from PIL import Image
import io
@api_view(["POST"])
def IdealWeight(data):

    body = json.loads(data.body)
   
    imageString = body.get("image")

    imgdata = base64.b64decode(str(imageString))
    image = Image.open(io.BytesIO(imgdata))
    img= cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
         classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
        
   
    return JsonResponse(""+ label , safe= False)