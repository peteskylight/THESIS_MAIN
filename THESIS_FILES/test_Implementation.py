import cv2
import argparse
import numpy as np
import os

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from tensorflow.keras.models import load_model
import tensorflow as tf

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

def detectMovements():
    return 0

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
    convertedModel = "actionsLITE.tflite"
    actionModel = tf.lite.Interpreter(model_path=convertedModel) # Load TFLite model and allocate tensors.
    actionModel.allocate_tensors()
    
    input_details = actionModel.get_input_details() #FOR CHECKING MODEL I/O DETAILS
    output_details = actionModel.get_output_details()

    actionsList = np.array(['Looking Left', 'Looking Right', 'Looking Up'])
    flattenedKeypoints = np.empty((3, 2), dtype=np.float64)
    sequence = []
    sentence = []
    recentAction = ''
    translateActionResult = ''

    
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
        
        img, humanResults = detectResults(frame, humanDetectorModel, 0.75)

        for result in humanResults:
            annotator = Annotator(frame)
            boxes = result.boxes
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
                            #PREPROCESS THE INPUTS
                            expandedKeypoints = np.expand_dims(sequence, axis=0)[0]
                            translatedExpanded = expandedKeypoints.astype(np.float32)
                            
                            #FEED THE INPUTS
                            actionModel.set_tensor(input_details[0]['index'], translatedExpanded)
                            
                            actionModel.invoke() #RUN INFERENCE
                            
                            actionResult = actionModel.get_tensor(output_details[0]['index'])
                            
                            translateActionResult = actionsList[np.argmax(actionResult)]
                            print(translateActionResult)
                            
                        except:
                            print(("="*10)+ "> > > PROBLEM HERE ! ! ! < < <"+("="*10))
                            continue

                    if recentAction != translateActionResult:
                        recentAction = translateActionResult

                except:
                    print("YOU MIGHT WANT TO CHECK IN HERE=======================<<<<<<<<")
                    continue

            drawBoundingBox(humanResults, img, recentAction)

            cv2.putText(img, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                     1.0, (255,255,255), 4, cv2.LINE_AA)
        
            cv2.imshow('Test Frame', img)

        
        if cv2.waitKey(10) == 27:
            break


if __name__ == "__main__":
    main()