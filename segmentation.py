from ultralytics import YOLO
import cv2
import torch
import time
from utils import *
from torchvision import models, transforms
from label_players import get_manual_ids, crop_frame, extract_embedding, get_corners

### Need for roboflow model basketball recognition
from inference_sdk import InferenceHTTPClient
import supervision as sv

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="zYBaqHuPPOScwarjQx8t" # free auto-generated api key from roboflow
)
###

player_team_map = {1: "A", 2: "A", 3: "A", 
                   4: "B", 5: "B", 6: "B"}

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    return transform

def get_resnet():
    resnet = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    return resnet

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

def get_frame_info(src):
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps

def id_from_emb(id_to_emb, box, frame, resnet, transform, conf = 0.75):
    crop = crop_frame(box, frame)
    emb = extract_embedding(crop, resnet, transform)

    best_id = None
    best_sim = -1

    for ref_id, ref_emb in id_to_emb.items():
        curr_sim = cos_sim(emb, ref_emb)
        if curr_sim > max(best_sim, conf):
            best_sim, best_id = curr_sim, ref_id
    
    return best_id

def draw_id_boxes(frame, tracked_ids):
    for box, id in tracked_ids:
        x1, y1, x2, y2 = get_corners(box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)    

def track_basketball(result):
    ### Roboflow model
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
    
def track_people(frame, result, records, id_to_emb, resnet, transform):
    ### YOLO model
    record = FrameRecord(result)
    records.append(record)

    annotated_frame = result.plot() # draw initial boxes w/ YOLO ids

    tracked_ids = []
    for box in record.person_boxes:
        new_id = id_from_emb(id_to_emb, box, frame, resnet, transform, 
                             conf = 0.9)

        if new_id:
            tracked_ids.append((box, new_id))

    draw_id_boxes(annotated_frame, tracked_ids) # overlay supervised IDs

    return annotated_frame

def create_annotated_replay(src, results, records, id_to_emb, resnet, transform):
    global TRACKING_BASKETBALL
    global TRACKING_PEOPLE

    print(f"Creating annotated replay at: {src}")
    frame_width, frame_height, vid_fps = get_frame_info(src)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    src_file = src.split('/')[-1]
    out = cv2.VideoWriter(f"./annotated_replays/labeled_{src_file}", 
                      fourcc = fourcc, fps = vid_fps, 
                      frameSize = (frame_width, frame_height))
    
    for frame_num, result in enumerate(results):
        if frame_num % 100 == 0: 
            print(f"Analyzing frame: {frame_num} ")

        frame = result.orig_img    

        if TRACKING_BASKETBALL: 
            annotated_frame = track_basketball(result)

        elif TRACKING_PEOPLE:
            annotated_frame = track_people(frame, result, records, id_to_emb,
                                           resnet, transform)
        
        out.write(annotated_frame)

    out.release()

def get_frame_count(src):
    cap = cv2.VideoCapture(src)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def main():
    records = []

    model = YOLO("./YOLO/yolo12n.pt")
    transform = get_transform()
    resnet = get_resnet()

    src = "./data_dir/raw_games/game_2_15s.MP4"

    frame_count = get_frame_count(src)
    print(f"There are {frame_count} frames in {src}.")

    id_to_emb = get_manual_ids(model, resnet, transform, src, 45)

    results = model.track(src, stream = True, conf = 0.6, 
                          tracker = "botsort.yaml", verbose = False)
    
    create_annotated_replay(src, results, records, id_to_emb, resnet, transform)

### Used for conditional compilation during development 
TRACKING_BASKETBALL = False 
TRACKING_PEOPLE = True
if __name__ == '__main__':
    main()