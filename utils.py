import torch.nn.functional as functional

def cos_sim(emb1, emb2):
    return functional.cosine_similarity(emb1, emb2, dim = 0).item()

def get_corners(box):
    return map(int, box.xyxy.squeeze().tolist())

def crop_frame(box, frame):
    x1, y1, x2, y2 = get_corners(box)
    return frame[y1:y2, x1:x2]

def get_person_boxes(result):
    boxes = result.boxes
    return [box for box in boxes if 
            result.names[int(box.cls)] == "person"]