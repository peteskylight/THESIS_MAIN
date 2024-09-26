import cv2
import numpy as np
import os
import argparse

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS

from camera_utils import Camera
from trackers import StudentTracker

def parse_arguments() -> argparse.Namespace: # For Camera
    parser=argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720], #default must be 1280, 720
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():

    # CAMERA PARAMETERS AREA
    # cameraInput = 'C:/Users/peter/Desktop/THESIS FILES/START/THESIS_MAIN/RESOURCES/VIDEOS/640 width/Left_Corner.mp4'
    cameraInput=0
    camera = cv2.VideoCapture(cameraInput)
    getFPS = CvFpsCalc(buffer_len=10)

    cameraObj = Camera()
    args = cameraObj.parse_arguments()
    frameWidth, frameHeight = args.webcam_resolution
    camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frameWidth, frameHeight))

    # MODEL AREA
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

            humanDetectorModel.drawBoundingBox(image, results, "TEST")
            cameraObj.drawFPSInfo(image, fps)
        
        # Write the frame to the output video file
        out.write(image)
        cv2.imshow("TEST", image)

        if cv2.waitKey(10) == 27:
            break

    # Release everything if job is finished
    camera.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# def main():

#     #CAMERA PARAMETERS AREA
#     cameraInput = 'C:/Users/peter/Desktop/THESIS FILES/START/THESIS_MAIN/RESOURCES/Video_DATASET/Room Keypoints/Left_Corner.mp4'
#     camera = cv2.VideoCapture(cameraInput)
#     getFPS = CvFpsCalc(buffer_len=10)

#     cameraObj = Camera()
#     args = cameraObj.parse_arguments()
#     frameWidth, frameHeight = args.webcam_resolution
#     camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

#     #MODEL AREA
#     humanDetectorModel = StudentTracker('yolov8n.pt')
#     humanPoseDetectorModel = StudentTracker('yolov8n-pose.pt')

#     while camera.isOpened():
#         fps = getFPS.get()

#         ret, frame = camera.read()

#         if not ret:
#             break
        
#         image, results = humanDetectorModel.trackHuman(frame=frame, confidenceRate=0.75)

        
#         for result in results:
#             annotator = Annotator(frame)
#             boxes = result.boxes
#             for box in boxes:
#                 b = box.xyxy[0]
#                 c = box.cls
#                 cropped_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

#                 try:
#                     _, poseResults = humanPoseDetectorModel.detectHumanPose(cropped_image, 0.7)
#                     flattenedKeypoints = humanPoseDetectorModel.drawLandmarks(cropped_image, poseResults)

#                 except Exception as e:
#                     print(e)

#             humanDetectorModel.drawBoundingBox(image,results, "TEST")
#             cameraObj.drawFPSInfo(image, fps)
#         cv2.imshow("TEST",image)

#         if cv2.waitKey(10) == 27:
#             break

