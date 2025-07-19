import torch.nn.functional as functional

def cos_sim(emb1, emb2):
    return functional.cosine_similarity(emb1, emb2, dim = 0).item() # [0,1]

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

def get_corners(box):
    return map(int, box.xyxy.squeeze().tolist())

def crop_frame(box, frame):
    x1, y1, x2, y2 = get_corners(box)
    return frame[y1:y2, x1:x2]

def get_person_boxes(result):
    boxes = result.boxes
    return [box for box in boxes if 
            result.names[int(box.cls)] == "person"]