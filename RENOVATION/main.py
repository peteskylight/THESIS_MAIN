import cv2
import numpy as np
import os
import argparse


from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils import CvFpsCalc #FOR FPS

from camera_utils import Camera
from utils import(VideoUtils,
                  DrawingUtils)

from trackers import StudentTracker
from trackers import RoomTracker


def getAbsPath(filePath):
    abs_path = os.path.abspath(filePath)
    print(abs_path)
    return abs_path

def main():
    
    #====================DIRECTORIES
    roomKeyDetectModel = 'RENOVATION/models/Room_Keypoints_YOLOv8_best.pt'
    student_detections_results = 'RENOVATION/cache/student_detection_results.pkl'
    student_pose_results = 'RENOVATION/cache/student_pose_results.pkl'
    
    #====================CREATE INSTANCES
    student_tracker = StudentTracker(humanDetectionModel='yolov8n.pt',
                                     humanDetectConf=0.5,
                                     humanPoseModel='yolov8n-pose.pt',
                                     humanPoseConf=0.5
                                     )

    roomAnnot_tracker = RoomTracker(roomAnnotationModel= getAbsPath(roomKeyDetectModel),
                                    roomAnnotConf=0.3)
    
    video_utils = VideoUtils()
    drawing_utils = DrawingUtils()
    
    #====================Read Video
    input_video_path = "C:/Users/peter/Desktop/THESIS FILES/START/THESIS_MAIN/RESOURCES/VIDEOS/Test/1920/Center.mp4"
    #input_video_path="C:/Users/peter/Desktop/THESIS FILES/START/Left_Corner.mp4"
    
    video_frames = video_utils.read_video(input_video_path)
    
    ##HEIGHT & WIDTH
    
    video_height = video_frames[0].shape[0]
    video_width = video_frames[0].shape[1]
    
    #====================Student Existence
    student_detections = student_tracker.detect_frames(frames=video_frames,
                                                       read_from_stub=False,
                                                       stub_path=student_detections_results
                                                       )
    
    #POSE ESTIMATION
    
    student_pose_results = student_tracker.detect_keypoints(frames=video_frames,
                                                            student_dicts=student_detections)
    #====================Classroom Keypoints Detection

    for frame, keypoints_dict, student_dict in zip(video_frames, student_pose_results, student_detections):
        frame_with_keypoints = drawing_utils.draw_keypoints(frame, keypoints_dict, student_dict)
        
        # Display the frame with keypoints
        cv2.imshow('Frame with Keypoints', frame_with_keypoints)
        cv2.waitKey(10)

    cv2.destroyAllWindows()
    
    room_detections = roomAnnot_tracker.detectRoomKeypoints(frames=video_frames)
    

    
    #Classroom Layout
    
    
    #Generate White Frames
    white_frames = []
    for frame in video_frames:
        generate = video_utils.generate_white_frame(height=video_height,
                                                        width=video_width)
        white_frames.append(generate)
    
    #DRAW OUTPUTS FOR WHITEFRAME
    output_white_frames = drawing_utils.draw_bboxes(video_frames = white_frames,
                                                    detections=student_detections,
                                                    )
    
    output_white_frames = drawing_utils.draw_room_bbox(detections=room_detections,
                                                       video_frames=output_white_frames)
    
    #Draw Outputs (FOR ORIGINAL VIDEO)
    output_video_frames = drawing_utils.draw_room_bbox(detections=room_detections,
                                         video_frames=video_frames)
    
    output_video_frames = drawing_utils.draw_bboxes(video_frames=output_video_frames,
                                                    detections=student_detections
                                                    )
    
    #EXPORT ORIGINAL VIDEO
    video_utils.save_video(output_video_frames=output_video_frames,
               output_video_path=getAbsPath("RENOVATION/output_videos/TESTORIG1.avi"),
               monitorFrames=False #Change me when everything is fuckedup
               )
    
    #EXPORT WHITE FRAMES
    video_utils.save_video(output_video_frames=output_white_frames,
               output_video_path=getAbsPath("RENOVATION/output_videos/TESTWHITE1.avi"),
               monitorFrames=False #Change me when everything is fuckedup
               )
if __name__ == "__main__":
    main()


