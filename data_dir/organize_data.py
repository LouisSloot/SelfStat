import os

### Script to restructure the MultiSubjects data splits in a clearer way
label_map = {
    'd': "dribble",
    'p': "layup",
    's': "shoot"
    }

def restructure():
    for split in ["train", "val", "test"]:
        for data_label in ["dribble", "shoot", "layup"]:
            dir_name = split + "_" + data_label
            os.mkdir(f"./train_val_test/{split}/{dir_name}")
        split_folder_path = f"./train_val_test/{split}"
        for video_name in os.listdir(split_folder_path):
            split_name = video_name.split("_") # "xxx_[d/s/p]_x.mp4 [0/1/2]"
            label_char = split_name[1]
            if label_char not in set(['d', 'p', 's']):
                continue
            video_label = label_map[label_char]
            src = f"./train_val_test/{split}/{video_name}"
            new_parent_folder = split + "_" + video_label
            dst = f"./train_val_test/{split}/{new_parent_folder}/{video_name}"
            os.rename(src = src, dst = dst)

if __name__ == "__main__":
    restructure()