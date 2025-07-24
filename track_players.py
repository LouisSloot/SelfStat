from reid_manager import IdentityManager
from label_players import get_sv_ids
from detect import YOLODetector
from create_replay import create_replay

def track_players():
    src = "./data_dir/raw_games/game_2_15s.MP4"
    FRAME_NUM = 65 # will eventually be input by the user
    num_players = 6 # testing with videos of 3v3 games

    id_manager = IdentityManager(num_players)
    model_path = "./YOLO/yolo12n.pt"
    detector = YOLODetector(model_path, conf = 0.5)
    
    sv_ids = get_sv_ids(detector, src, FRAME_NUM)

    crops = [pair[1] for pair in sv_ids]
    id_manager.build_embedding_refs(crops)

    results = detector.detect_video(src)
    create_replay(src, results, id_manager)

def main():
    track_players()

if __name__ == '__main__':
    main()