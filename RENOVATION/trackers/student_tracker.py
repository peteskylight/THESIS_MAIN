from ultralytics import YOLO
import cv2
import pickle
import numpy as np
import sys
sys.path.append('../')

from ultralytics.utils.plotting import Annotator

class StudentTracker:
    def __init__(self, humanDetectionModel, humanDetectConf, humanPoseModel, humanPoseConf):
        self.humanDetectModel = YOLO(humanDetectionModel)
        self.humanPoseModel = YOLO(humanPoseModel)
        self.humanDetectConf = humanDetectConf
        self.humanPoseConf = humanPoseConf
    
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        student_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                student_detections = pickle.load(f)
            return student_detections

        for frame in frames:
            student_dict = self.trackHuman(frame)
            student_detections.append(student_dict)
            # cv2.imshow("TEST", frame)
            # cv2.waitKey(10)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(student_detections, f)
        
        return student_detections

    def detectHumanPose(self, frame, confidenceRate):
        # Recolor Feed from RGB to BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Perform inference using the YOLO model
        results = self.humanPoseModel(frame, conf = self.humanPoseConf)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, results
        
        
    def trackHuman(self, frame, confidenceRate=0.3):
        # Perform inference using the YOLO model
        results = self.humanDetectModel.track(frame, conf = self.humanDetectConf, persist=True, classes=0)[0]
        id_name_dict = results.names
        
        student_dict = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                #Get the image per person
                b = box.xyxy[0]
                c = box.cls
                cropped_image = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                
                try:     
                    _, poseResults = self.detectHumanPose(cropped_image, 0.3)
                    flattenedKeypoints = self.drawLandmarks(cropped_image, poseResults)
                    #Track ID
                    
                except Exception as e:
                    print(e)
                    
                if box.id is not None and box.xyxy is not None and box.cls is not None:
                    track_id = int(box.id.tolist()[0])
                    track_result = box.xyxy.tolist()[0]
                    object_cls_id = box.cls.tolist()[0]
                    object_cls_name = id_name_dict.get(object_cls_id, "unknown")
                    if object_cls_name == "person":
                        student_dict[track_id] = track_result
                else:
                    print("One of the attributes is None:", box.id, box.xyxy, box.cls)
        return student_dict

    
    def drawLandmarks(self, image, poseResults):
        keypoints_normalized = np.array(poseResults[0].keypoints.xyn.cpu().numpy()[0])
        flattenedKeypoints = keypoints_normalized.flatten()
        
        for keypointsResults in keypoints_normalized:
            x = keypointsResults[0]
            y = keypointsResults[1]
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])),
                                    3, (0, 255, 0), -1)

    
    