from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, conf = 0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(source = frame, stream = False,
                                          conf = self.conf)
        boxes = results[0]
        return boxes