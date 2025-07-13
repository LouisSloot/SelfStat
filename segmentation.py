from ultralytics import YOLO
import cv2
from label_players import get_manual_ids

### Needed to use roboflow model for basketball recognition
from inference_sdk import InferenceHTTPClient
import supervision as sv

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="zYBaqHuPPOScwarjQx8t" # free auto-generated api key from roboflow
)
###

player_team_map = {1: "A", 2: "A", 3: "A", 
                   4: "B", 5: "B", 6: "B"}

def ball_found(ball_box):
    """ Determines whether or not there is a bounding box to be drawn for 
        the basektball. """
    return bool(ball_box.xyxy.tolist())

def find_nearest_player(ball_box, person_boxes):
    """ Used to find the player currently in possession of the basketball.
        Returns None if no player appears to have possession (loose ball).
        Current naive approach relies solely on IoU. """
    if not ball_found(ball_box): return None
    best_IOU = 0
    best_person = None
    for person_box in person_boxes:
        curr_IOU = findIOU(ball_box.xyxy.tolist()[0], 
                           person_box.xyxy.tolist()[0])
        if curr_IOU > best_IOU:
            best_IOU, best_person = curr_IOU, person_box
    return best_person

def findIOU(xyxy1, xyxy2):
    """ Standard Intersection over Union function. """
    left1, top1, right1, btm1 = xyxy1
    left2, top2, right2, btm2 = xyxy2
    
    intersection_w = min(right1, right2) - max(left1, left2)
    intersection_h = min(btm1, btm2) - max(top1, top2)

    intersection = intersection_w * intersection_h
    if intersection <= 0: return 0

    area1 = (right1 - left1) * (btm1 - top1)
    area2 = (right2 - left2) * (btm2 - top2)
    union = area1 + area2 - intersection # don't double-count overlap

    return intersection / union

class FrameRecord():
    """ Data structure used to store metadata for each frame. """
    frame_idx = 1
    def __init__(self, result):
        boxes = result.boxes
        person_mask = [result.names[cls.item()] == "person" for cls in 
                       boxes.cls]
        ball_mask = [result.names[cls.item()] == "sports ball" for cls in 
                     boxes.cls]
        
        self.ball_box = boxes[ball_mask]
        self.person_boxes = boxes[person_mask]

        self.player_possession = find_nearest_player(self.ball_box, 
                                                     self.person_boxes)
        self.team_possession = player_team_map.get(self.player_possession, None)

        self.frame_idx = FrameRecord.frame_idx
        FrameRecord.frame_idx += 1

### Write out annotated frames to a video for me to watch back for testing
def get_frame_info(src):
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps

def track_basketball(result):
    img = result.orig_img
    result_basketball = CLIENT.infer(img,
                                model_id="basketball-player-detection-v8kcy/6")
    detections = sv.Detections.from_inference(result_basketball)
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()   
    annotated_frame = bounding_box_annotator.annotate(scene = img, 
                                                    detections = detections)
    annotated_frame = label_annotator.annotate(scene = annotated_frame, 
                                            detections = detections)
    return annotated_frame
    
def track_people(result, records):
    record = FrameRecord(result)
    records.append(record)
    result.boxes = record.player_possession
    annotated_frame = result.plot()
    return annotated_frame

def create_annotated_replay(src, src_file, results, records, TRACKING_BASKETBALL,
                            TRACKING_PEOPLE):
    frame_width, frame_height, vid_fps = get_frame_info(src)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"./annotated_replays/labeled_{src_file}", 
                      fourcc = fourcc, fps = vid_fps, 
                      frameSize = (frame_width, frame_height))

    for frame_num, result in enumerate(results):

        if TRACKING_BASKETBALL: 
            annotated_frame = track_basketball(result)

        elif TRACKING_PEOPLE:
            annotated_frame = track_people(result, records)
            
        out.write(annotated_frame)

    out.release()

def main():
    records = []

    model = YOLO("./yolo12n.pt")

    src_root = "./data_dir/raw_games/"
    src_file = "game_2_30s.MP4"
    src = src_root + src_file

    id_to_emb = get_manual_ids(model, src, 765) # again, 765 is a magic frame for testing

    results = model.track(src, stream = True, conf = 0.4, verbose = False)

    TRACKING_BASKETBALL = False # used for essentially conditional compilation during development 
    TRACKING_PEOPLE = True
    
    create_annotated_replay(src, src_file, results, records, 
                            TRACKING_BASKETBALL, TRACKING_PEOPLE)

if __name__ == '__main__':
    main()