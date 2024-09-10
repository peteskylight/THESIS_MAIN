from ultralytics import YOLO
import cv2
import numpy as np

from ultralytics.utils.plotting import Annotator

class StudentTracker:
    def __init__(self, model):
        self.model = YOLO(model)
    
    def trackHuman(self, frame, confidenceRate):
        # Recolor Feed from RGB to BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Perform inference using the YOLO model
        results = self.model.track(frame, conf = confidenceRate, persist=True, )

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, results

    def detectHumanPose(self, frame, confidenceRate):
            # Recolor Feed from RGB to BGR
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Perform inference using the YOLO model
            results = self.model(frame, conf = confidenceRate)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image, results

    def drawBoundingBox(self, frame, results, action):
        for result in results:
            annotator = Annotator(frame)
            boxes = result.boxes
            # SHOW FPS
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, action)
        return frame
    
    def drawLandmarks(self, image, poseResults):
        keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
        flattenedKeypoints = keypoints_normalized.flatten()

        #print(flattenedList)
        for keypointsResults in keypoints_normalized:
            x = keypointsResults[0]
            y = keypointsResults[1]
            #print("X: {} | Y: {}".format(x,y))
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])),
                                    3, (0, 255, 0), -1)
        
        return flattenedKeypoints
    
    