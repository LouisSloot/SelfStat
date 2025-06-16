import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import cv2
import numpy as np
from pathlib import Path


class BasketballDataset(Dataset):

    def __init__(self, annotation_file, video_root, transform=None, clip_len=32):
        self.video_root = Path(video_root)
        self.clip_len = clip_len
        self.transform = transform
        
        # load annotations
        self.annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        # remove 'videos/' prefix
                        video_path = parts[0].replace('videos/', '')
                        label = int(parts[1])
                        self.annotations.append((video_path, label))
        
        print(f"Loaded {len(self.annotations)} videos from {annotation_file}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        video_path, label = self.annotations[idx]
        full_path = self.video_root / video_path
        
        # load video frames
        frames = self.load_video(full_path)
        
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label
    
    def load_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) > self.clip_len: # uniform sampling
            indices = np.linspace(0, len(frames) - 1, self.clip_len, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.clip_len:
            while len(frames) < self.clip_len:
                frames.extend(frames)
            frames = frames[:self.clip_len]
        
        # convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames = np.array(frames)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # C, T, H, W
        
        return frames

class VideoTransform:
    
    def __init__(self, size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
    
    def __call__(self, video):
        # video: (C, T, H, W)
        C, T, H, W = video.shape
        
        # resize frames
        video = video.permute(1, 0, 2, 3)  # T, C, H, W
        resized_frames = []
        
        for i in range(T):
            frame = video[i]  # C, H, W
            frame = transforms.functional.resize(frame, self.size)
            resized_frames.append(frame)
        
        video = torch.stack(resized_frames)  # T, C, H, W
        video = video.permute(1, 0, 2, 3)  # C, T, H, W
        
        # normalize
        video = video / 255.0
        video = (video - self.mean) / self.std
        
        return video