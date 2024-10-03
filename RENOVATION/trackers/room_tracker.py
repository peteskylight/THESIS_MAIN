from ultralytics import YOLO
import cv2
import numpy as np

class RoomTracker:
    def __init__(self, roomAnnotationModel, roomAnnotConf):
        self.roomAnnotationModel = YOLO(roomAnnotationModel)
        self.roomAnnotConf = roomAnnotConf
    
    def detectRoomKeypoints(self, frames):
        room_detections = []
        
        for frame in frames:
            results = self.roomAnnotationModel.predict(frame, conf=self.roomAnnotConf, save=False)
            for result in results:
                per_frame = []
                detected_classes = set()
                boxes = result.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    class_name = self.roomAnnotationModel.names[int(c)]
                    
                    #Limit to 1 detection of class
                    if class_name not in detected_classes:
                        per_frame.append({'box': b.tolist(), 'class': class_name})
                        detected_classes.add(class_name)
                
                room_detections.append(per_frame)
        
        return room_detections
                
def detectRoomKeypoints(self, frames):
    room_detections = []
    
    for frame in frames:
        results = self.roomAnnotationModel.predict(frame, conf=self.roomAnnotConf, save=False)
        for result in results:
            per_frame = []
            detected_classes = set()
            boxes = result.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                class_name = self.roomAnnotationModel.names[int(c)]
                
                if class_name not in detected_classes:
                    per_frame.append({'box': b.tolist(), 'class': class_name})
                    detected_classes.add(class_name)
                    
            room_detections.append(per_frame)
    
    return room_detections



