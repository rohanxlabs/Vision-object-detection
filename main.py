import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ROOT_DIR)
import cv2 
from camera.camera import Camera
from detection.model import ObjectDetectionModel
from detection.detector import Detector
from utils.draw import draw_boxes
from utils.fps import FPS
from config.config import *

def main():
    camera = Camera(CAMERA_INDEX)
    model = ObjectDetectionModel(MODEL_PATH)
    detector = Detector(model)
    fps_counter = FPS()

    cv2.namedWindow("Vision Object Detection",
cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vision Object Detection",
800, 600)

    while True:
        ret , frame = camera.read()
        if not ret:
            print("camera not return frame")
            break

        detections = detector.detect(frame)
        frame = draw_boxes(frame, detections)
        fps = fps_counter.calculate()
        cv2.putText(frame, f"FPS: {fps}",(20,40),cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255),2)
        cv2.imshow("Vision Object Detection" , frame)
        key = cv2.waitKey(30)
        if key == 27:  # ESC key to exit
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

