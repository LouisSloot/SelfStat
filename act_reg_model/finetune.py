import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
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
from data_prep import VideoDataset, VideoTransform


def train(model, train_loader, val_loader, device, epochs = 10, lr = 1e-5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    for epoch in range(epochs):
        print(f'Beginning epoch {epoch}/{epochs}:')
        print('-' * 20)

def load_model(num_classes): # will try to add more stats (classes) over time
    model = r2plus1d_18(pretrained = True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
