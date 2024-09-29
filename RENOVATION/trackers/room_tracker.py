from ultralytics import YOLO
import cv2
import numpy as np

class RoomTracker:
    def __init__(self, roomAnnotationModel, roomAnnotConf):
        self.roomAnnotationModel = YOLO(roomAnnotationModel)
        self.roomAnnotConf = roomAnnotConf
    
    def detectRoomKeypoints(self, frames):
        room_detections = []
        per_frame = []
        
        for frame in frames:
            results = self.roomAnnotationModel.predict(frame, conf = self.roomAnnotConf, save=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    per_frame.append(b.tolist())
                    print(c.tolist())
                room_detections.append(per_frame)
                per_frame=[]
        #print(room_detections)
        return room_detections
                
                

