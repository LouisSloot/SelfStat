import cv2
from utils import *

def draw_unlabeled_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = get_corners(box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

def on_mouse_press(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        box_to_remove = -1 # default value indicating no box was clicked

        for i, box in enumerate(param["unlabeled_boxes"]):
            x1, y1, x2, y2 = get_corners(box)
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"Clicked on box at ({x},{y})")
                player_id = input("Enter player ID: ")
                ### TODO: Add validity checks for the entered ID here
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

def run_user_labeling(annotated_frame, unlabeled_boxes, param):
    draw_unlabeled_boxes(annotated_frame, unlabeled_boxes)
    cv2.imshow("Label Frame", annotated_frame)
    cv2.setMouseCallback("Label Frame", on_mouse_press, param = param)
    print("Click on each player you want to assign an ID to.")

    #TODO: change the condition that closes this window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_sv_ids(detector, vid_src, frame_num):
    """ Return a list of tuples (id, crop) pairing the user-labeled crops
        of the user-chosen reference frame with the respective user-entered IDs. """
    frame = get_frame_from_vid(vid_src, frame_num)
    # TODO: Account for if frame is None here

    result = detector.detect_frame(frame)
    unlabeled_boxes = get_person_boxes(result) 
    selected_boxes, str_ids = [], []
    annotated_frame = frame.copy()

    param_map = {
        "unlabeled_boxes": unlabeled_boxes,
        "selected_boxes": selected_boxes,
        "manual_ids": str_ids,
        "frame": annotated_frame
    }

    run_user_labeling(annotated_frame, unlabeled_boxes, param_map)

    crops = [crop_frame(box, frame) for box in selected_boxes]
    sv_ids = zip(str_ids, crops)

    return sv_ids


def main():
   return

if __name__ == '__main__':
    main()