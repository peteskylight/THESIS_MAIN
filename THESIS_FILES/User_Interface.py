import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk 

import argparse

import cv2

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

def open_camera(): 
	# REPLACE OR COMMENT THIS IF GLOBAL IMAGE IS GOING TO BE USED
    _, frame = camera.read() 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    resized_frame = cv2.resize(image, (640,350))

    
    captured_image = Image.fromarray(resized_frame) # Capture the latest frame and transform to image 

    photo_image = ImageTk.PhotoImage(image=captured_image) # Convert captured image to photoimage 
    
    label_widget.photo_image = photo_image # Displaying photoimage in the label 

    # Configure image in the label 
    label_widget.configure(image=photo_image) 

    # Repeat the same process after every 10 seconds 
    label_widget.after(10, open_camera) 

def setWindowParameters(window):
    #Setting in center of the screen
    screenWidth = window.winfo_screenwidth()
    screenHeight = window.winfo_screenheight()
    x = (screenWidth - window.winfo_reqwidth()) // 3.5
    y = (screenHeight - window.winfo_reqheight()) // 10
    
    #Lock the window dimensions
    window.resizable(height=None, width=None)
    
    #Set window size & window coordinates
    window.geometry(f"700x600+{int(x)}+{int(y)}")



def main():
    #GLOBAL VARIABLES
    global camera
    global label_widget
    
    #Main Window
    root = tk.Tk()

    root.bind('<Escape>', lambda e: root.quit()) 
    
    setWindowParameters(root)
    tk.Button(root, text="Quit", command=root.destroy).place(x=650, y=0)
    
    #Camera
    args = parse_arguments()
    frameWidth, frameHeight = args.webcam_resolution
    cameraInput = 0
    camera = cv2.VideoCapture(cameraInput)  # Use the specified camera
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    
    label_widget = Label(root) 
    label_widget.pack() 
    
    button1 = Button(root, text="Open Camera", command=open_camera)
    button1.pack() 
    
    #Start GUI Event Loop
    root.mainloop()

if __name__ == "__main__":
    main()