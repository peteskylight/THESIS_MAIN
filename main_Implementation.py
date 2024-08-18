import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS

cameraInput = 'students4.jpg'

humanDetectorModel = YOLO('yolov8n.pt')
humanPoseDetectorModel = YOLO('yolov8n-pose.pt')
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
    humanResults = humanDetectorModel(frame, conf = 0.3 , classes = 0)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for result in humanResults:
        
        annotator = Annotator(frame)
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            cropped_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            
            try:
                poseResults = humanPoseDetectorModel(cropped_image, conf = 0.7, classes = 0)
                keypoints_normalized = poseResults[0].keypoints.xyn.cpu().numpy()[0]
                print(keypoints_normalized)
                annotator.box_label(b, "Human")
            except:
                continue
            
        cv2.imshow('Annotated Frame', frame)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    if cv2.waitKey(10000) == 27:
        break