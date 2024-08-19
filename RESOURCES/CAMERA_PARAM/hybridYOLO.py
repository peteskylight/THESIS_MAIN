import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pydantic import BaseModel
import argparse
import csv
from ultralytics.utils.plotting import Annotator
from model import KeyPointClassifier
from model import PointHistoryClassifier

model = YOLO("yolov8n-pose.pt")
keypoint_classifier = KeyPointClassifier()

# Load keypoint classifier labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

def identifyPose():
    results = model(frame)
    annotator = Annotator(frame)
    # Process results and draw bounding boxes and pose classification
    for result in results:
    
    #     boxes = result.boxes
    #     for box in boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0]
        keypoints_normalized = result.keypoints.xyn.cpu().numpy()[0]
        preProcessedLandmarkList = keypoints_normalized.flatten().tolist()
        poseID = keypoint_classifier(preProcessedLandmarkList)
        label = keypoint_classifier_labels[poseID] if poseID != 5 else "None"
        # Draw bounding box and pose classification
        annotator.box_label([xmin, ymin, xmax, ymax], label)
    
            # Display the annotated image using OpenCV
    cv2.imshow('Annotated Frame', annotator.result())

# Initialize YOLOv8 model
yolo_model = YOLO("yolov8n-pose.pt")
yolo_model.classes = [0]  # Set the class to detect people

cap = cv2.VideoCapture(0)  # Use the default camera (usually index 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Run YOLOv8 inference
    result = yolo_model(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for r in result:
        annotator = Annotator(frame)
        boxes = r.boxes
        counter = 0
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0]
            keypoints_normalized = r.keypoints.xyn.cpu().numpy()[0]
            preProcessedLandmarkList = keypoints_normalized.flatten().tolist()
            poseID = keypoint_classifier(preProcessedLandmarkList)
            label = keypoint_classifier_labels[poseID] if poseID != 5 else "None"
            # Draw bounding box and pose classification
            annotator.box_label([xmin, ymin, xmax, ymax], label)
            # Display the annotated image using OpenCV
            counter += 1
            print("=========>>>> PERSON#{}; POSITION: {}". format(counter, label))
            cv2.imshow('Annotated Frame', annotator.result())
        preProcessedLandmarkList=[]
            
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
