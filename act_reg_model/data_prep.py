import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_video
import random

# I use "label" to mean "dribble", "layup", or "shoot"
# I use "class" to mean 0, 1, or 2

### Define a pytorch Dataset class for MultiSubjects videos
class VideoDataset(Dataset):
    def __init__(self, annotation_file, data_root, split, clip_frames = 63,
                 transform = None):
        self.data_root = data_root
        self.clip_frames = clip_frames # 63 frames = ~2.5s at 25 fps
        self.transform = transform
        self.split = split # "train", "val", or "test"
        self.annotations = []

        # build out list of self.annotations
        f = open(annotation_file, 'r')
        for line in f: # "{video_file} [0/1/2]"
            line = line.strip()
            if line:
                parts = line.split()
                video_name = parts[0]
                video_class = int(parts[1])
                self.annotations.append((video_name, video_class))
        f.close()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        video_name, video_class = self.annotations[index]
        video_label = video_label_from_name(video_name)
        sub_dir = self.split + "_" + video_label
        full_path = f'{self.data_root}/{sub_dir}/{video_name}'

        video, _, _ = read_video(full_path, pts_unit = "sec", 
                                 output_format = "CTHW")
        
        # trim/pad video to the correct number of frames
        if video.shape[1] > self.clip_frames:
            start = random.randint(0, video.shape[1] - self.clip_frames)
            video = video[:, start:start + self.clip_frames, :, :]

        elif video.shape[1] < self.clip_frames:
            pad = self.frames_per_clip - video.shape[1]
            video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, pad))
        
        if self.transform: # will always be true (in this script at least)
            video = self.transform(video)
        
        return video, video_class

    def augment_video(self, video):
        ### Artificially modifying videos to diversify dataset

        C, T, H, W = video.shape

        # horizontal flip
        if random.random() < 0.5:
            video = torch.flip(video, dims = [3])

        # brightness and contrast
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            video = video * brightness_factor
            video = torch.clamp(video, 0, 255)


### Transform class essentially used as a function to preprocess videos
class VideoTransform():
    def __init__(self, size = (224, 224), mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225]): # ImageNet standard values
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, video):
        C, T, H, W = video.shape

        video = video.permute(1, 0, 2, 3) # put T in first index (T,C,H,W)
        resized_frames = []

        # loop over and resize frames
        for i in range(T):
            frame = video[i] # in C,H,W format
            frame = transforms.functional.resize(frame, self.size)
            resized_frames.append(frame)

        video = torch.stack(resized_frames)
        video = video.permute(1, 0, 2, 3)
        ### Permutation matrices are their own inverse :D ^^

        video = video / 255.0
        video = (video - self.mean) / self.std

        return video

    
def video_label_from_name(video_name):
    # video_name should be as appears in annotation file
    label_map = {
        'd': "dribble",
        'p': "layup",
        's': "shoot"
    }
    parts = video_name.split("_")
    video_label = label_map[parts[1]]
    return video_label