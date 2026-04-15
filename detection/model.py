from ultralytics import YOLO

class ObjectDetectionModel:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def predict(self, frame):
        return self.model(frame, stream=True)