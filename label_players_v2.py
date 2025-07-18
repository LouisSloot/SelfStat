from detect import YOLODetector
from utils import *
import cv2

detector = YOLODetector(model_path = "./YOLO/yolo12n.pt")
id_manager = IdentityManager() # eventually put these where ...IDs is called from

def assign_and_draw_IDs(annotated_frame):
    
    boxes = detector.detect(annotated_frame)

    boxes = get_person_boxes(boxes)

    for box in boxes:

        crop = crop_frame(annotated_frame, box)
        pID = id_manager.identify(crop)

        if pID is not None:
            x1, y1, x2, y2 = get_corners(box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"Player: {pID}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))