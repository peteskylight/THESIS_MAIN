import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS

#import torch #=========================================================> GPU IMPORTANT <========
#torch.cuda.set_device(0) 

cameraInput = 0

#REPLACE MO NITO BOSS YUNG NASA BABA
#humanDetectorModel = YOLO('yolov8n.pt', task='detect').to('cuda')
#humanPoseDetectorModel = YOLO('yolov8n-pose.pt', task='detect').to('cuda')

humanDetectorModel = YOLO('yolov8n.pt') #Icomment mo na lan to boss
humanPoseDetectorModel = YOLO('yolov8n-pose.pt')# Pati ito

humanDetectorModel.classes = [0] #Limit to human detection
humanPoseDetectorModel.classes = [0] #Limit to juman detection

camera = cv2.VideoCapture(cameraInput)

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
    
    # Perform inference using the YOLO model
    poseResults = humanPoseDetectorModel(frame,show=True, conf = 0.7)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for result in poseResults:
        annotator = Annotator(frame)
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            cropped_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            keypoints_normalized = poseResults[0].keypoints.xyn.cpu().numpy()[0]
            print(keypoints_normalized)
            for person in keypoints_normalized:
                for x, y, conf in person:
                    if conf > 0.5:  # Only draw keypoints with high confidence
                        cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 3, (0, 255, 0), -1)
            annotator.box_label(b, "Tite ni Bennett")
            
        cv2.putText(frame, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow('Annotated Frame', image)
    
    if cv2.waitKey(10) == 27:
        break