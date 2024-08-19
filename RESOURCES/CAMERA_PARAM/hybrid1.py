import cv2
import mediapipe as mp
from ultralytics import YOLO

# Initialize YOLOv8 model
yolo_model = YOLO("yolov8n-pose.pt")
yolo_model.classes = [0]  # Set the class to detect people

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

cap = cv2.VideoCapture(0)  # Use the default camera (usually index 0)

mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Run YOLOv8 inference
    result = yolo_model(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for r in result:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
            c = box.cls

            # Run MediaPipe pose estimation on the cropped region
            cropped_image = image[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            results = pose.process(cropped_image)

            # Draw landmarks on the original image
            mp_drawing.draw_landmarks(cropped_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.imshow('YOLO V8 Detection with Pose Estimation', image)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
