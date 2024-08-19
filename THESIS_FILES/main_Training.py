import cv2
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.image as img

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS

#import torch #========================================> GPU IMPORTANT <========


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

def drawLandmarks(image, x, y):
    cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])),
                                   3, (0, 255, 0), -1)
    return image

def folderSetUp():
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('./THESIS_MAIN', 'THESIS_FILES', 'HumanPose_DATA') 

    # Actions that we try to detect
    actions = np.array(['Looking Right', 'Looking Left', 'Looking Up'])

    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30
    
    #========> ACTUAL MAKING DIRECTORIES <========
    for action in actions: 
        for sequence in range(no_sequences+1):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    

def main():

    cameraInput = 0

    args = parse_arguments()
    frameWidth, frameHeight = args.webcam_resolution
    camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)


    #=====================================================> UNCOMMENT THIS FOR GPU <=====
    #torch.cuda.set_device(0) 
    #humanDetectorModel = YOLO('yolov8n.pt', task='detect').to('cuda')
    #humanPoseDetectorModel = YOLO('yolov8n-pose.pt', task='detect').to('cuda')

    humanDetectorModel = YOLO('yolov8n.pt') #COMMENT THIS AND (V)THIS(V) when GPU
    humanPoseDetectorModel = YOLO('yolov8n-pose.pt')# <==========THIS

    humanDetectorModel.classes = [0] #Limit to human detection
    humanPoseDetectorModel.classes = [0] #Limit to juman detection

    getFPS = CvFpsCalc(buffer_len=10)

    while True:
        # Read a frame from the camera
        fps = getFPS.get()
        ret, frame = camera.read()
        if not ret:
            break
        # Recolor Feed from RGB to BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        
        #Higher confidence but needs good lighting & low frame drop
        
        poseResults = humanPoseDetectorModel(frame, conf = 0.75)
        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        k = cv2.waitKey(10)

        for result in poseResults:
            annotator = Annotator(frame)
            boxes = result.boxes

            # SHOW FPS
            cv2.putText(image, "FPS: " + str(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255,255,255), 4, cv2.LINE_AA) 
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls

                try:
                    keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
                    
                    flattenedKeypoints = keypoints_normalized.flatten()
                    flattenedList = flattenedKeypoints.tolist()
                    
                    print(flattenedList)
                    for keypointsResults in keypoints_normalized:
                        x = keypointsResults[0]
                        y = keypointsResults[1]
                        #print("X: {} | Y: {}".format(x,y))
                        drawLandmarks(image, x, y)

                        
                except:
                    print("NO PERSON DETECTED!")
                    continue
                annotator.box_label(b, "Human Subject")

                #Key Pressing
                
                #if k == 102: #Letter "f" for frames
                
                if k == 114: #Letter "r" for recording
                    cv2.putText(image, "FPS: " + str(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0,0,0), 4, cv2.LINE_AA)
        cv2.imshow("Window", image)

        if k == 27:
            break 

if __name__ == "__main__":
    folderSetUp()