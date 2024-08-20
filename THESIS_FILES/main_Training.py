import cv2
import numpy as np
import argparse
import os
import time

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

def drawLandmarks(image, poseResults):
    keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
                    
    flattenedKeypoints = keypoints_normalized.flatten()
    flattenedList = flattenedKeypoints.tolist()
    #print(flattenedList)
    for keypointsResults in keypoints_normalized:
        x = keypointsResults[0]
        y = keypointsResults[1]
        #print("X: {} | Y: {}".format(x,y))
        cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])),
                                   3, (0, 255, 0), -1)
    drawBoundingBox(poseResults, image)

    return flattenedKeypoints


def folderSetUp(DATA_PATH, actionsList, noOfSequences):
    #========> ACTUAL MAKING DIRECTORIES <========
    for action in actionsList: 
        for sequence in range(1, noOfSequences+1):
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

def drawBoundingBox(poseResults, frame):
    for result in poseResults:
        annotator = Annotator(frame)
        boxes = result.boxes
        # SHOW FPS
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, "Human Subject")

def loggingKeypoints(camera, frame, mode, poseResults,
                     DATA_PATH, image, actionsList, startFolder,
                     noOfSequences, sequenceLength, getFPS):
    font = cv2.FONT_HERSHEY_SIMPLEX

    if mode == 0:
        for result in poseResults:
            annotator = Annotator(frame)
            boxes = result.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                # SHOW FPS
                
                try:
                    drawLandmarks(image, poseResults)
                except:
                     print("NO PERSON DETECTED!")
                     continue
                annotator.box_label(b, "Human Subject")

    if mode == 1:
        # camera.release()
        # cv2.destroyAllWindows()

        # NEW LOOP
        for action in actionsList:
            # Loop through sequences aka videos
            for sequence in range(1, startFolder+1):
                # Loop through video length aka sequence length
                for frame_num in range(1, sequenceLength+1):
                    # Read feed
                    
                    ret, frame = camera.read()
                    img, results = detectPose(frame, 0.75)

                    flattenedList = drawLandmarks(img, results)

                    # NEW Apply wait logic
                    if frame_num == 1: 
                        
                        # Show to screen
                        start_time = time.time()
                        while ((time.time() - start_time) < 2):
                            # Continue reading frames to avoid blocking
                            ret, frame = camera.read()
                            mode = 2
                            displayWithInfo(frame, mode, camera, getFPS, action, sequence, (time.time() - start_time))
                        mode = 1
                    else:
                        displayWithInfo(img, mode, camera, getFPS, action, sequence, 0)
                        
                    # NEW Export keypoints
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, flattenedList)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        mode = 0
    return image

def main():

    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('THESIS_FILES', 'HumanPose_DATA') 

    # Actions that we try to detect
    actionsList = np.array(['Looking Left', 'Looking Right', 'Looking Up'])

    # Thirty videos worth of data
    noOfSequences = 30

    # Videos are going to be 30 frames in length
    sequenceLength = 30
    startFolder = 30

    folderSetUp(DATA_PATH, actionsList, noOfSequences)

    cameraInput = 0
    getFPS = CvFpsCalc(buffer_len=10)

    args = parse_arguments()
    frameWidth, frameHeight = args.webcam_resolution
    camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)


    #=====================================================> UNCOMMENT THIS FOR GPU <=====
    #torch.cuda.set_device(0) 
    #humanDetectorModel = YOLO('yolov8n.pt', task='detect').to('cuda')
    #humanPoseDetectorModel = YOLO('yolov8n-pose.pt', task='detect').to('cuda')

    humanDetectorModel = YOLO('yolov8n.pt') #COMMENT THIS AND (V)THIS(V) when GPU
    humanPoseDetectorModel = YOLO('yolov8n-pose.pt')# <==========THIS
    #=> ^^^COMMENT THESE TWO FOR GPU ^^^ <=
    #====================================================================================
    humanDetectorModel.classes = [0] #Limit to human detection
    humanPoseDetectorModel.classes = [0] #Limit to juman detection

    mode = 0

    while True:
        key = cv2.waitKey(10)
        mode = selectMode(key, mode)
    
        # Read a frame from the camera
        
        ret, frame = camera.read()

        if not ret:
            break

        image, poseResults = detectPose(frame, 0.75)

        image = loggingKeypoints(camera, frame, mode, poseResults,
                     DATA_PATH, image, actionsList, startFolder,
                     noOfSequences, sequenceLength, getFPS)

        displayWithInfo(image, mode, camera, getFPS, action=0, sequence=0, timeRemaining=0)
        

    camera.release()
    cv2.destroyAllWindows()
        

def selectMode(key, mode):
    if key == 110:  # "n" for NORMAL MODE
        mode = 0
    if key == 114:  # "r" for RECORDING MODE
        mode = 1
    return mode

def displayWithInfo(image, mode, camera, getFPS, action, sequence, timeRemaining):
    start_time = time.time()

    if cv2.waitKey(10)== 27:
        camera.release()
        cv2.destroyAllWindows() 

    font = cv2.FONT_HERSHEY_SIMPLEX


    fps = getFPS.get()

    cv2.putText(image, "FPS: " + str(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255,255,255), 4, cv2.LINE_AA)
    if mode == 0:
        cv2.putText(image, "Mode: Normal (0)", (10, 70),
                    font, 1.0, (255,255,255), 4, cv2.LINE_AA)
    if mode == 1:
        cv2.putText(image, "Mode: Recording (1)", (10, 70),
                    font, 1.0, (255,255,255), 4, cv2.LINE_AA)
        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (10,110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    if mode == 2:
        cv2.putText(image, 'STARTING COLLECTION', (150,220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 6, cv2.LINE_AA)
        cv2.putText(image, '{} : # {}'.format(action, sequence), (180,250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
        cv2.putText(image, 'in {}'.format(round(2-timeRemaining, 2)), (250,280),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 6, cv2.LINE_AA)
    cv2.imshow("Window Original", image)

if __name__ == "__main__":
    main()