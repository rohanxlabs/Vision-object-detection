from config.config import CONFIDENCE_THRESHOLD 

class Detector:
    def __init__(self, model):
        self.model = model

    def detect(self, frame):
        detections = []

        for result in self.model.predict(frame):
         for box in result.boxes:
            if box.conf[0] >= CONFIDENCE_THRESHOLD:
               detections.append({
                  "xyxy":box.xyxy[0].tolist(),
                  "confidence":float(box.conf[0]),
                  "class_id":int(box.cls[0]),
                  "class_name":result.names[int(box.cls[0])]
               })
        return detections