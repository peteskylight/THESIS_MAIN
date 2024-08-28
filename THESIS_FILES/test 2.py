import cv2
from PIL import Image, ImageTk
from tkinter import Label

def open_camera(): 
    # REPLACE OR COMMENT THIS IF GLOBAL IMAGE IS GOING TO BE USED
    _, frame = camera.read() 

    # Resize the frame to desired dimensions (e.g., 640x480)
    resized_frame = cv2.resize(frame, (640, 480))

    # Convert image from one color space to another 
    opencv_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGBA) 

    # Capture the latest frame and transform to image 
    captured_image = Image.fromarray(opencv_image) 

    # Convert captured image to photoimage 
    photo_image = ImageTk.PhotoImage(image=captured_image) 

    # Displaying photoimage in the label 
    label_widget.photo_image = photo_image 

    # Configure image in the label 
    label_widget.configure(image=photo_image) 

    # Repeat the same process after every 10 milliseconds 
    label_widget.after(10, open_camera)

# Example usage
camera = cv2.VideoCapture(0)
root = Tk()
label_widget = Label(root)
label_widget.pack()
open_camera()
root.mainloop()
