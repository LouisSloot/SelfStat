from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, conf = 0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_video(self, src):
        results = self.model.predict(source = src, stream = True,
                                          conf = self.conf,
                                          verbose = False)
        return results
    
    def detect_frame(self, frame):
        results = self.model.predict(source = frame, stream = False, 
                                     conf = self.conf,
                                     verbose = False)
        return results[0]