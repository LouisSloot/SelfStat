import cv2
from utils import *

def draw_labeled_boxes(frame, box_to_pID):
    for box, pID in box_to_pID:
        x1, y1, x2, y2 = get_corners(box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {pID}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)    

def create_replay(src, results, id_manager):
    print(f"Creating annotated replay of: {src}")
    dest = f"./annotated_replays/labeled_{src_file}"

    frame_width, frame_height, vid_fps = get_vid_info(src)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    src_file = src.split('/')[-1]
    out = cv2.VideoWriter(dest, fourcc = fourcc, fps = vid_fps, 
                          frameSize = (frame_width, frame_height))
    
    frame_count = get_frame_count(src)

    for frame_idx, result in enumerate(results):

        if frame_idx % 100 == 0: print(f"Frame {frame_idx} / {frame_count}")

        frame = result.orig_img
        annotated_frame = frame.copy()

        person_boxes = get_person_boxes(result)

        box_to_pID = id_manager.identify(person_boxes, frame)

        draw_labeled_boxes(annotated_frame, box_to_pID)

        out.write(annotated_frame)