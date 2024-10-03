import numpy as np
import os
import cv2

def generate_white_frame(height, width):
    # Create a white frame (all pixel values set to 255)
    white_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    return white_frame

def getAbsPath(filePath):
    abs_path = os.path.abspath(filePath)
    return abs_path

def frames_to_video(frames, output_path, fps=30):
    if not frames:
        raise ValueError("The frames array is empty.")
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_path}")

# Example usage
height = 480
width = 640
num_frames = 1000

# Generate white frames and store them in an array
frames_array = [generate_white_frame(height, width) for _ in range(num_frames)]

# Convert the array of frames to a video
output_video_path = getAbsPath('RENOVATION/output_videos/white_frames_video.mp4')
frames_to_video(frames_array, output_video_path)
