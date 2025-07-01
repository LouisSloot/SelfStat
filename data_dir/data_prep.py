import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import argparse
import json
import random


### Define a pytorch Dataset class for MultiSubjects videos
class VideoDataset(Dataset):
    def __init__(self, annotation_file, data_dir, clip_frames = 63,
                 transform = None):
        self.data_dir = data_dir
        self.clip_frames = clip_frames # 63 frames = ~2.5s at 25 fps
