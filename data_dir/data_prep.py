import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
from torchvision.io import read_video
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import argparse
import json
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
        pass

    
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