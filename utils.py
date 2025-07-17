def get_corners(box):
    return map(int, box.xyxy.squeeze().tolist())

def crop_frame(box, frame):
    x1, y1, x2, y2 = get_corners(box)
    return frame[y1:y2, x1:x2]