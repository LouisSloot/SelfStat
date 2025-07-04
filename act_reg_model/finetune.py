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
    
    best_acc = 0

    for epoch in range(epochs):
        print(f'Beginning epoch {epoch+1}/{epochs}:')
        print('-' * 20)

        train_acc, train_loss = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        val_acc, val_loss = val_epoch(model, val_loader, criterion, device)

        print(f"Train accuracy: {train_acc:.2f}%, Train loss: {train_loss:.2f}")
        print(f"Val accuracy: {val_acc:.2f}%, Val loss: {val_loss:.2f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, 
            './best_models/best_model.pth')
            print(f"New best model -- Accuracy: {best_acc:.2f}%")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    curr_loss = 0.0
    correct = 0
    total = 0

    for input, label in train_loader:
        input, label = input.to(device), label.to(device)
        output = model(input)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_loss += loss.item()
        _, label_guess = torch.max(output, 1) # top-1 accuracy
        total += label.size(0)
        correct_tensor = (label_guess == label)
        correct += correct_tensor.sum().item()
    
    acc = 100 * correct / total
    return acc, curr_loss


def val_epoch(model, val_loader, criterion, device):
    model.eval()
    curr_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input, label in val_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            loss = criterion(output)

            curr_loss += loss.item()
            _, label_guess = torch.max(output, 1)
            total += label.size(0)
            correct_tensor = (label_guess == label)
            correct += correct_tensor.sum().item()
    
    acc = 100 * correct / total
    return acc, curr_loss


def load_model(num_classes): # will try to add more stats (classes) over time
    model = r2plus1d_18(pretrained = True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    return

if __name__ == '__main__':
    main()