import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from utils import CvFpsCalc  # FOR FPS

# Initialize models
humanDetectorModel = YOLO('yolov8n.pt')
humanPoseDetectorModel = YOLO('yolov8n-pose.pt')

humanDetectorModel.classes = [0]  # Limit to human detection
humanPoseDetectorModel.classes = [0]  # Limit to human detection

cameraInput = 0
camera = cv2.VideoCapture(cameraInput)
getFPS = CvFpsCalc(buffer_len=10)

batch_size = 1  # Define your batch size
frames = []

while True:
    fps = getFPS.get()
    ret, frame = camera.read()
    if not ret:
        break

    frames.append(frame)
    if len(frames) == batch_size:
        # Perform batch inference
        poseResults = humanPoseDetectorModel(frames, conf=0.8)

        for i, frame in enumerate(frames):
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            result = poseResults[i]
            annotator = Annotator(frame)
            boxes = result.boxes

            for box in boxes:
                k = cv2.waitKey(10)
                b = box.xyxy[0]
                c = box.cls
                cropped_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                keypoints_normalized = np.array(result.keypoints.xyn.cpu().numpy()[0])

                for keypointsResults in keypoints_normalized:
                    x = keypointsResults[0]
                    y = keypointsResults[1]
                    print("X: {} | Y: {}".format(x, y))

                    #cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 3, (0, 255, 0), -1)

                #annotator.box_label(b, "Human Subject")
    
                
                #---------NOTING FOR KEY PRESSING USING CV2.WAITKEY
                # if k == 102:  # Letter "f" for frames
                #     cv2.putText(image, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #                 1.0, (255, 255, 255), 4, cv2.LINE_AA)
                # if k == 114:  # Letter "r" for recording
                #     cv2.putText(image, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #                 1.0, (0, 0, 0), 4, cv2.LINE_AA)

            
            # cv2.putText(image, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #                             1.0, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.imshow("Window", image)
        frames = []  # Clear the batch

    
    if cv2.waitKey(10) == 27:
        break

camera.release()
cv2.destroyAllWindows()
