import cv2
import numpy as np
import os
import argparse

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS

from camera_utils import Camera
from trackers import StudentTracker

def main():

    #CAMERA PARAMETERS AREA
    cameraInput = 1
    camera = cv2.VideoCapture(cameraInput)
    getFPS = CvFpsCalc(buffer_len=10)

    cameraObj = Camera()
    args = cameraObj.parse_arguments()
    frameWidth, frameHeight = args.webcam_resolution
    camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

    #MODEL AREA
    humanDetectorModel = StudentTracker('yolov8n.pt')
    humanPoseDetectorModel = StudentTracker('yolov8n-pose.pt')

    while camera.isOpened():
        fps = getFPS.get()

        ret, frame = camera.read()

        if not ret:
            break
        
        image, results = humanDetectorModel.trackHuman(frame=frame, confidenceRate=0.75)

        
        for result in results:
            annotator = Annotator(frame)
            boxes = result.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                cropped_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

                try:
                    _, poseResults = humanPoseDetectorModel.detectHumanPose(cropped_image, 0.7)
                    flattenedKeypoints = humanPoseDetectorModel.drawLandmarks(cropped_image, poseResults)

                except Exception as e:
                    print(e)

            humanDetectorModel.drawBoundingBox(image,results, "TEST")
            cameraObj.drawFPSInfo(image, fps)
        cv2.imshow("TEST",image)

        if cv2.waitKey(10) == 27:
            break


if __name__ == "__main__":
    main()
