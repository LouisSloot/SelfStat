import cv2
import torch
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np

#TODO: check that extracting the embeddings is actually helpful in persistent
#      IDs. also make reassigning IDs not overlap names, and put some limit on
#      how many ids can be assigned based on a number given by the user ... this
#      is me thinking ahead to when there are background people with boxes whom
#      the program and user will naturally want to ignore.

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    return transform

def get_resnet():
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    return resnet

def crop_frame(box, frame):
    x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
    return frame[y1:y2, x1:x2]

def extract_embedding(crop):
    transform = get_transform()
    resnet = get_resnet()
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        return resnet(tensor).squeeze(0)

def draw_unlabeled_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

def on_mouse_press(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        box_to_remove = -1 # default value indicating no box was clicked
        for i, box in enumerate(param["unlabeled_boxes"]):
            x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"Clicked on box at ({x},{y})")
                player_id = input("Enter player ID: ")
                param["manual_ids"].append(player_id)
                param["selected_boxes"].append(box)
                box_to_remove = i
                cv2.rectangle(param["frame"], (x1, y1), 
                              (x2, y2), (0, 255, 0), 2)
                cv2.putText(param["frame"], f"ID {player_id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Label Frame", param["frame"])
        if box_to_remove > -1:
            param["unlabeled_boxes"].pop(box_to_remove)

def make_embeddings_map(ids, boxes, frame):
    id_to_emb = dict()
    for id, box in zip(ids, boxes):
        crop = crop_frame(box, frame)
        emb = extract_embedding(crop)
        id_to_emb[id] = emb
    return id_to_emb

def get_frame_from_vid(vid_src, frame_num):
    cap = cv2.VideoCapture(vid_src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    cap.release()
    if success:
        return frame
    else:
        print(f"Failed to get frame {frame_num}.")
        return get_frame_from_vid(vid_src, 0) # default return first frame
        
def get_person_boxes(model, frame):
    results = model.predict(frame, conf = 0.6, verbose = False, show = False)
    result = results[0] # always only one frame
    return [box for box in result.boxes if 
            result.names[int(box.cls)] == "person"]

def run_user_labeling(annotated_frame, unlabeled_boxes, param):
    draw_unlabeled_boxes(annotated_frame, unlabeled_boxes)
    cv2.imshow("Label Frame", annotated_frame)
    cv2.setMouseCallback("Label Frame", on_mouse_press, param = param)
    print("Click on each player you want to assign an ID to.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_manual_ids(model, vid_src, frame_num):

    frame = get_frame_from_vid(vid_src, frame_num)

    unlabeled_boxes = get_person_boxes(model, frame) 
    selected_boxes = []
    manual_ids = []
    annotated_frame = frame.copy()

    param_map = {
        "unlabeled_boxes": unlabeled_boxes,
        "selected_boxes": selected_boxes,
        "manual_ids": manual_ids,
        "frame": annotated_frame
    }

    run_user_labeling(annotated_frame, unlabeled_boxes, param_map)

    id_to_emb = make_embeddings_map(manual_ids, selected_boxes, frame)
    return id_to_emb


def main():
    src = "./data_dir/raw_games/game_2_30s.MP4"
    FRAME_TO_LABEL = 765 # hand-picked for testing right now
    model = YOLO("./yolo12n.pt")
    id_to_emb = get_manual_ids(model, src, FRAME_TO_LABEL)

if __name__ == '__main__':
    main()