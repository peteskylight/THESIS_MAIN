import cv2
import numpy as np
import os
import argparse
import os

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS

from camera_utils import Camera
from utils import(read_video,
                  save_video,
                  draw_bboxes,
                  draw_room_bbox)
    
from trackers import StudentTracker
from trackers import RoomTracker


def getAbsPath(filePath):
    abs_path = os.path.abspath(filePath)
    return abs_path

def main():
    # Read Video
    input_video_path = "C:/Users/peter/Desktop/THESIS FILES/START/THESIS_MAIN/RESOURCES/VIDEOS/Test/640width/1.mp4"
    #input_video_path="C:/Users/peter/Desktop/THESIS FILES/START/Left_Corner.mp4"
    
    video_frames = read_video(input_video_path)
    
    #Create Instances
    # student_tracker = StudentTracker(humanDetectionModel='yolov8n.pt',
    #                                  humanDetectConf=0.4,
    #                                  humanPoseModel='yolov8n-pose.pt',
    #                                  humanPoseConf=0.4
    #                                  )
    roomKeyDetectModel = 'RENOVATION/models/Room_Keypoints_YOLOv8_best.pt'

    
    roomAnnot_tracker = RoomTracker(roomAnnotationModel= getAbsPath(roomKeyDetectModel),
                                    roomAnnotConf=0.3)
    
    #Student Existence & Pose Detection
    # student_detections = student_tracker.detect_frames(frames=video_frames,
    #                                                    read_from_stub=False,
    #                                                    stub_path=None
    #                                                    )

    #Classroom Keypoints Detection
    
    room_detections = roomAnnot_tracker.detectRoomKeypoints(frames=video_frames)
    
    #Classroom Layout
    
    #Draw Outputs
    output_video_frames = draw_room_bbox(detections=room_detections,
                                         video_frames=video_frames)
    # output_video_frames = draw_bboxes(video_frames=video_frames,
    #                 student_detections=student_detections
    #                                                   )

    save_video(output_video_frames=output_video_frames,
               output_video_path="try4.avi",
               monitorFrames=True #Change me when everything is fuckedup
               )
    
if __name__ == "__main__":
    main()

