from reid_manager import IdentityManager
from label_players import supervised_label
from detect import YOLODetector
from create_replay import create_replay

def track_players():
    src = "./data_dir/raw_games/game_2_good_start.MP4"
    FRAME_NUM = 1 # will eventually be input by the user
    num_players = 6 # testing with videos of 3v3 games

    model_path = "./YOLO/yolo12n.pt"
    detector = YOLODetector(model_path, conf = 0.5)

    sv_ids, crops, selected_boxes = supervised_label(detector, src, FRAME_NUM)
    id_manager = IdentityManager(num_players, sv_ids)
    id_manager.build_embedding_refs(crops)
    id_manager.build_last_pos(selected_boxes)

    results = detector.detect_video(src)
    create_replay(src, results, id_manager)

def main():
    track_players()

if __name__ == '__main__':
    main()