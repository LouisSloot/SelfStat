import torch.nn.functional as functional
import cv2

def cos_sim(emb1, emb2):
    """ Returns the cosine similarity of two tensors. """
    return functional.cosine_similarity(emb1, emb2, dim = 0).item() # [-1,1]

def findIOU(xyxy1, xyxy2):
    """ Find the Intersection over Union of two rectangular boxes. """
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
    """ Return the integer coordinates of the top left and bottom right 
        corners of a bounding box. """
    return map(int, box.xyxy.squeeze().tolist())

def crop_frame(box, frame):
    """ Crops a video frame/photo down to the given bounding box. """
    x1, y1, x2, y2 = get_corners(box)
    return frame[y1:y2, x1:x2]

def get_person_boxes(result):
    """ Returns only the bounding boxes detected for people from a YOLO results
        object. """
    boxes = result.boxes
    return [box for box in boxes if 
            result.names[int(box.cls)] == "person"]

def normalize(x, x_min, x_max):
    """ Min - Max normalization. """
    return (x - x_min) / (x_max - x_min) # maps to [0,1]

def get_vid_info(src):
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps

def get_frame_count(src):
    cap = cv2.VideoCapture(src)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def get_frame_from_vid(vid_src, frame_num):
    cap = cv2.VideoCapture(vid_src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    cap.release()
    if success:
        return frame
    else:
        print(f"Failed to get frame {frame_num}. Returning None.")
        return None