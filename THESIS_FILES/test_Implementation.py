#ESSENTIAL AREA
import cv2
import argparse
import numpy as np
import os
import sys
import time
sys.path.append('deep_sort/deep/checkpoint/ckpt.t7')

#OBJECT DETECTION AREA
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

#ACTION RECIGNITION AREA
from tensorflow.keras.models import load_model

#FRAME RATE AREA
from utils import CvFpsCalc #FOR FPS

#DEEPSORT AREA - JUST USE PIP INSTALL
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

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
    
    return flattenedKeypoints

def drawBoundingBox(poseResults, frame, action):
    for result in poseResults:
        annotator = Annotator(frame)
        boxes = result.boxes
        # SHOW FPS
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, action)

def detectResults(frame,model, confidenceRate):
    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Perform inference using the YOLO model
    results = model(frame, conf = confidenceRate)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results



def main():

    #YOLO AREA

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

    #TENSORFLOW AREA
    model = "actions.h5"
    actionModel = load_model(model)

    actionsList = np.array(['Looking Left', 'Looking Right', 'Looking Up'])
    flattenedKeypoints = np.empty((3, 2), dtype=np.float64)
    sequence = []
    sentence = []
    recentAction = ''
    translateActionResult = ''

    #DEEPSORT AREA

    deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
    tracker = DeepSort(model_path=deep_sort_weights, max_age=70)
    frames = []
    unique_track_ids = set()
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    i = 0
    counter, fps2, elapsed = 0, 0, 0
    start_time = time.perf_counter()
    
    #PARAMETERS AREA
    cameraInput = 0
    camera = cv2.VideoCapture(cameraInput)
    getFPS = CvFpsCalc(buffer_len=10)

    args = parse_arguments()
    frameWidth, frameHeight = args.webcam_resolution
    camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)


    while camera.isOpened():
        # Read a frame from the camera
        fps = getFPS.get()

        ret, frame = camera.read()
        
        if not ret:
            break
        
        img, humanResults = detectResults(frame, humanDetectorModel, 0.7)

        for result in humanResults:
            annotator = Annotator(frame)
            boxes = result.boxes
            clsList = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh  # box with xywh format, (N, 4)
            for class_index in clsList:
                class_name = class_names[int(class_index)]
                #print("Class:", class_name)
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                cropped_image = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                try:
                    poseResults = humanPoseDetectorModel(cropped_image, conf = 0.7)
                    flattenedKeypoints =  drawLandmarks(image=cropped_image, poseResults=poseResults)
                    sequence.append(flattenedKeypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        try:
                            actionResult = actionModel.predict(np.expand_dims(sequence, axis=0))[0]
                            translateActionResult = actionsList[np.argmax(actionResult)]
                            print(translateActionResult)
                        except Exception as e:
                            print(("="*10)+ "> > > PROBLEM @ MODEL! {} < < <".format(e))
                            continue

                    if recentAction != translateActionResult:
                        recentAction = translateActionResult

                except:
                    print("YOU MIGHT WANT TO CHECK IN HERE=======================<<<<<<<<")
                    continue
            #drawBoundingBox(humanResults, img, recentAction)
            cv2.putText(img, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                     1.0, (255,255,255), 4, cv2.LINE_AA)
            
        red_cls = np.array(clsList)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)
        
        tracks = tracker.update(bboxes_xywh, conf, img)
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # Set color values for red, blue, and green
            red_color = (0, 0, 255)  # (B, G, R)
            blue_color = (255, 0, 0)  # (B, G, R)
            green_color = (0, 255, 0)  # (B, G, R)

            # Determine color based on track_id
            color_id = track_id % 3
            if color_id == 0:
                color = red_color
            elif color_id == 1:
                color = blue_color
            else:
                color = green_color

            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(img, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            # Add the track_id to the set of unique track IDs
            unique_track_ids.add(track_id)

        # Update the person count based on the number of unique track IDs
        person_count = len(unique_track_ids)

        # Update FPS and place on frame
        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps2 = counter / elapsed
            counter = 0
            start_time = current_time

        # Draw person count on frame
        cv2.putText(img, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Append the frame to the list
        frames.append(img)

        cv2.imshow('Test Frame', img)
        
        if cv2.waitKey(10) == 27:
            break


if __name__ == "__main__":
    main()