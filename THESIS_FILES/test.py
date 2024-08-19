import cv2
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.image as img

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS

#For checking purposes

# Initialize models
humanDetectorModel = YOLO('yolov8n.pt')
humanPoseDetectorModel = YOLO('yolov8n-pose.pt')

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

def drawLandmarks(image, x, y, poseResults):
    keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
                    
    flattenedKeypoints = keypoints_normalized.flatten()
    flattenedList = flattenedKeypoints.tolist()
    
    print(flattenedList)
    for keypointsResults in keypoints_normalized:
        x = keypointsResults[0]
        y = keypointsResults[1]
        #print("X: {} | Y: {}".format(x,y))
    cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])),
                                   3, (0, 255, 0), -1)
    return image

def setGlobal():
    global DATA_PATH, start_folder
    global actionsList
    global no_sequences
    global sequence_length

def folderSetUp():
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('./THESIS_MAIN', 'THESIS_FILES', 'HumanPose_DATA') 

    # Actions that we try to detect
    actionsList = np.array(['Looking Right', 'Looking Left', 'Looking Up'])

    # Thirty videos worth of data
    no_sequences = 30

    # Videos are going to be 30 frames in length
    sequence_length = 30
    start_folder = 30
    
    #========> ACTUAL MAKING DIRECTORIES <========
    for action in actionsList: 
        for sequence in range(no_sequences+1):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    
def detectPose(frame, confidenceRate):
    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    #Higher confidence but needs good lighting & low frame drop
    poseResults = humanPoseDetectorModel(frame, conf = confidenceRate)
    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, poseResults

def drawBoundingBox(poseResults, frame, fps):
    for result in poseResults:
        annotator = Annotator(frame)
        boxes = result.boxes

        # SHOW FPS
        cv2.putText(frame, "FPS: " + str(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255,255,255), 4, cv2.LINE_AA) 
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls

def loggingKeypoints(cap, frame, mode, fps, poseResults):
    if mode == 0:
        for result in poseResults:
            annotator = Annotator(frame)
            boxes = result.boxes

            # SHOW FPS
            cv2.putText(frame, "FPS: " + str(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255,255,255), 4, cv2.LINE_AA) 
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls

                try:
                    drawLandmarks(frame, x, y, poseResults)
                except:
                    print("NO PERSON DETECTED!")
                    continue
                annotator.box_label(b, "Human Subject")

    if mode == 1:
        # NEW LOOP
        # Loop through actions
        for action in actionsList:
            # Loop through sequences aka videos
            for sequence in range(start_folder, start_folder+no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()

                    keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
                    
                    flattenedKeypoints = keypoints_normalized.flatten()
                    flattenedList = flattenedKeypoints.tolist()
                    
                    print(flattenedList)
                    for keypointsResults in keypoints_normalized:
                        x = keypointsResults[0]
                        y = keypointsResults[1]
                        #print("X: {} | Y: {}".format(x,y))
                        drawLandmarks(frame, x, y)

                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', frame)
                    
                    # NEW Export keypoints
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, flattenedList)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break


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


    mode = 0


    while True:
        # Read a frame from the camera
        fps = getFPS.get()
        ret, frame = camera.read()
        if not ret:
            break

        key = cv2.waitKey(10)
        mode = selectMode(key, mode)

        image, poseResults = detectPose(frame, 0.75)

        

                #Key Pressing
                
                #if key == 102: #Letter "f" for frames
                
                if key == 114: #Letter "r" for recording
                    cv2.putText(image, "FPS: " + str(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0,0,0), 4, cv2.LINE_AA)
        cv2.imshow("Window", image)

        if key == 27:
            break 

def selectMode(key, mode):
    if key == 110:  # "n" for NORMAL MODE
        mode = 0
    if key == 114:  # "r" for RECORDING MODE
        mode = 1
    return mode

if __name__ == "__main__":
    folderSetUp()