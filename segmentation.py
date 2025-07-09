from ultralytics import YOLO

player_team_map = {1: "A", 2: "A", 3: "A", 
                   4: "B", 5: "B", 6: "B"}

def find_nearest_player(ball_box, person_boxes):
    """ Used to find the player currently in possession of the basketball.
        Returns None if no player appears to have possession (loose ball).
        Current naive approach relies solely on IoU. """
    if not ball_box: return None
    best_IOU = 0
    best_person = None
    for person_box in person_boxes:
        curr_IOU = findIOU(ball_box.xyxy.tolist()[0], 
                           person_box.xyxy.tolist()[0])
        if curr_IOU > best_IOU:
            best_IOU, best_person = curr_IOU, person_box
    return best_person

def findIOU(xyxy1, xyxy2):
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

        self.frame_idx = FrameRecord.frame_idx
        FrameRecord.frame_idx += 1
        # self.ball_box, self.ball_in_frame = (ball_box, True if ball_box else 
                                            #  None, False)
        self.player_possession = find_nearest_player(self.ball_box, 
                                                     self.person_boxes)
        self.team_possession = player_team_map.get(self.player_possession, None)

model = YOLO("./yolo11m.pt")

src = "./data_dir/raw_games/game_1.mp4"

results = model.predict(src, stream = True, conf = 0.4)

keep_classes = {"sports ball", "person"}

for frame_num, result in enumerate(results):

    if frame_num > 40: break

    record = FrameRecord(result)

    result.boxes = record.player_possession # determine which box(es) to draw

    result.show()  # display to screen