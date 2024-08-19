import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pydantic import BaseModel
import argparse
import csv

from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS
from model import KeyPointClassifier
from model import PointHistoryClassifier



cameraInput = 0
csv_path = 'keypointsSample.csv'

def parse_arguments() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720], #default must be 1280, 720
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def select_mode(key, mode):
    number = -1
    
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
        print("================K IS PRESSED!!!!!!=================")
    # if key == 104:  # h
    #     mode = 2
    return number, mode

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        print('-------------------------------------------IM HERE')
    # if mode == 2 and (0 <= number <= 9):
    #     csv_path = 'model/point_history_classifier/point_history.csv'
    #     with open(csv_path, 'a', newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([number, *point_history_list])
    return

def main():
    #---------INITIALIZE----------
    keypoint_classifier = KeyPointClassifier()
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    

    # Initialize camera capture
    args = parse_arguments()
    frameWidth, frameHeight = args.webcam_resolution
    camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    
    # Initialize YOLO-POSE model
    model = YOLO("yolov8n-pose.pt")
    model2 = YOLO("yolov8n.pt")
    model.classes = [0] #=======> CLASSIFY ONLY PEOPLE
    
    
    # ============= CSV FOR LABELS ============
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    
    #-----------------MAIN LOOP -------------
    while True:
        # Read a frame from the camera
        ret, frame = camera.read()
        if not ret:
            break
        # Recolor Feed from RGB to BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Perform inference using the YOLO model
        results = model2(frame, conf = 0.75 , classes = 0)
        
        counter = 1
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
        # Process results and draw bounding boxes and pose classification
        for result in results:
            fps = cvFpsCalc.get()
            
            annotator = Annotator(frame)
            boxes = result.boxes
            counter += 1
            print("===========PERSON # {}".format(counter))
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                cropped_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                
                try:
                    results2 = model(cropped_image, conf = 0.8, classes = 0)
                    keypoints_normalized = results2[0].keypoints.xyn.cpu().numpy()[0]
                    preProcessedLandmarkList = []
                    for sublist in keypoints_normalized: #=========> FOR REARRANGING THE LIST
                        for landmark in sublist:
                            preProcessedLandmarkList.append(landmark)
                    poseID = keypoint_classifier(preProcessedLandmarkList)
                    label = keypoint_classifier_labels[poseID] if poseID != 5 else "None"
                    # Draw bounding box and pose classification
                    annotator.box_label(b, label)
                except:
                    continue
                # Display the annotated image using OpenCV  
        
            cv2.imshow('Annotated Frame', annotator.result())
        print("===========PERSON # {}".format(counter))
        
                
        # Exit loop if ESC is pressed
        if cv2.waitKey(10) == 27:
            break
    
    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()