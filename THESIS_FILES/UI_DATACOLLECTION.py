from tkinter import *
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk

import argparse
import os

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

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

def setWindowParameters(window):
    #Setting in center of the screen
    screenWidth = window.winfo_screenwidth()
    screenHeight = window.winfo_screenheight()
    x = (screenWidth - window.winfo_reqwidth()) // 3.9
    y = (screenHeight - window.winfo_reqheight()) // 10
    
    #Lock the window dimensions
    window.resizable(False, False)
    
    #Set window size & window coordinates
    window.geometry(f"900x600+{int(x)}+{int(y)}")

def detectHuman(frame, humanPoseDetectorModel, confidenceRate):
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    #Higher confidence but needs good lighting & low frame drop
    poseResults = humanPoseDetectorModel(frame, conf = confidenceRate)
    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, poseResults
        
def execDetect():
    global isDetect
    if isDetect:
        isDetect = False
        detectButton.config(text='Start Detect')
    else:
        isDetect =  True
        detectButton.config(text='Stop Detect')

def addAction():
    noOfSequences = 30
    DATA_PATH = os.path.join('THESIS_FILES', 'HumanPose_DATA') 
    actionsList = []
    actionInput = folderNameInput.get(1.0, "end-1c")
    flagExist = False

    if actionInput == '':
        messagebox.showinfo("Information", "Please enter action name!")
    else:
        actionsList.append(actionInput)
        for action in actionsList:
            for sequence in range(1, noOfSequences+1):
                try: 
                    os.makedirs(os.path.join(DATA_PATH, action,
                                            str(sequence)))
                except FileExistsError:
                    flagExist = True
                    pass
        if flagExist:
            messagebox.showerror(title="Already Existed!", message="The name of action you entered already exists!")
        folderNameInput.delete("1.0", "end")

def drawUIelements(root):
    #GLOBAL AREA
    global quitButton
    global openCameraButton
    global detectButton
    global folderNameInput
    
    #Title
    title = Label(root, text="A.R.M.S. Data Collection", anchor=tk.CENTER, height=1, width=45,
                  font=("Courier New", 25, "bold")).pack()
    
    #QUIT BUTTON
    quitButton = Button(root, text="Quit", command=root.destroy).place(relx=0.93, rely=0.01)
    
    #OPEN CAMERA BUTTON
    openCameraButton = Button(root, text="Open Camera", command=showCamera)
    openCameraButton.place(relx=0.1, rely=0.92)
    
    #DETECT BUTTON
    detectButton = Button(root, text="Start Detecting", command=execDetect)
    detectButton.place(relx=0.2, rely=0.92)
    detectButton.config(state=DISABLED)
    
    #TEXTBOX
    folderNameInput = Text(root, height=1, width=20)
    folderNameInput.place(relx=0.3, rely=0.92)
    
    #MAKE FOLDER BUTTON
    makeFolder = Button(root, text="Add Action", command=addAction)
    makeFolder.place(relx=0.5, rely=0.92)


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


def showCamera(): 
    global isDetect
    
	# REPLACE OR COMMENT THIS IF GLOBAL IMAGE IS GOING TO BE USED
    _, frame = camera.read() 
    
    if isDetect:
        image, poseResults = detectHuman(frame, humanPoseDetectorModel, 0.75)
        for result in poseResults:
            annotator = Annotator(image)
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

        
        convertImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        
    else:
        convertImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    
    
    resized_frame = cv2.resize(convertImage, (640,480))

    
    captured_image = Image.fromarray(resized_frame) # Capture the latest frame and transform to image 
    photo_image = ImageTk.PhotoImage(image=captured_image) # Convert captured image to photoimage 
    cameraLabel.photo_image = photo_image # Displaying photoimage in the label 

    # Configure image in the label 
    cameraLabel.configure(image=photo_image) 
    openCameraButton.config(state=tk.DISABLED)
    detectButton.config(state=NORMAL)
    # Repeat the same process after every 10 seconds 
    cameraLabel.after(10, showCamera) 

def main():
    
    #GLOBAL VARIABLES AREA
    global isDetect
    global cameraLabel
    global camera
    global humanDetectorModel
    global humanPoseDetectorModel
    
    isDetect = False
    
    #MAIN WINDOW ARE
    root = tk.Tk()
    root.bind('<Escape>', lambda e: root.quit())
    
    #CAMERA PARAMETERS AREA
    args = parse_arguments()
    frameWidth, frameHeight = args.webcam_resolution
    cameraInput = 0
    camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    
    
    
    #ACTION RECOGNITION VARIABLES
    
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
    
    #WIDGET VARIABLES AREA
    cameraLabel = Label(root)
    cameraLabel.place(relx=0.5, rely=0.5, anchor="center")
    
    
    #FUNCTIONS AREA
    setWindowParameters(root)
    drawUIelements(root)
    
    root.mainloop()


if __name__ == "__main__":
    main()