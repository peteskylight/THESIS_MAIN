import argparse
import cv2

class Camera:
    def __init__(self):
        return

    def parse_arguments(self) -> argparse.Namespace: # For Camera
        self.parser = argparse.ArgumentParser(description="YOLOv8 Live")
        self.parser=argparse.ArgumentParser(description="YOLOv8 Live")
        self.parser.add_argument(
            "--webcam-resolution",
            default=[1280,720], #default must be 1280, 720
            nargs=2,
            type=int
        )
        args = self.parser.parse_args()
        return args
    
    def drawFPSInfo(self, img, fps):
        cv2.putText(img, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                     1.0, (255,255,255), 4, cv2.LINE_AA)