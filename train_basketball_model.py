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
    
class Simplified3DCNN(nn.Module):
    
    def __init__(self, num_classes=3, dropout_rate = 0.5):
        super(Simplified3DCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 3D conv block #1 -- 16 filters
            nn.Conv3d(3, 16, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # 3D conv block #2 -- 32 filters
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # 3D conv block #3 -- 64 filters
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # dropout to reduce overfitting
        self.dropout1 = nn.Dropout(p = 0.3)
        self.dropout2 = nn.Dropout(p = dropout_rate)

        # small classfier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            self.dropout2,
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)

        if self.training:
            x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                       max_norm=1.0) # gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval() # disables dropout
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
    
    return total_loss / len(dataloader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Basketball Action Recognition Training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate') # lower learning rate to avoid overfitting
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (L2 regularization)')
    parser.add_argument('--data-root', default='data/multisubjects', help='Dataset root')
    parser.add_argument('--work-dir', default='./work_dirs/simple_train', help='Working directory')
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    
    device = torch.device('cpu')  # no NVIDIA GPU : (
    print(f"Using device: {device}")
    
    transform = VideoTransform()
    
    train_dataset = BasketballDataset(
        annotation_file=f"{args.data_root}/train_list.txt",
        video_root=f"{args.data_root}/videos",
        transform=transform
    )
    
    val_dataset = BasketballDataset(
        annotation_file=f"{args.data_root}/val_list.txt",
        video_root=f"{args.data_root}/videos",
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=False
    )
    
    model = Simplified3DCNN(num_classes = 3, dropout_rate = args.dropout).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, 
                            weight_decay = 0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor = 0.5, 
                                                     patience = 3)
    
    best_acc = 0
    train_history = []
    val_history = []
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print("-" * 50)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # val
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        train_history.append({'loss': train_loss, 'acc': train_acc})
        val_history.append({'loss': val_loss, 'acc': val_acc})
        
        # epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'val_history': val_history,
            }, work_dir / 'best_model.pth')
            print(f"New best model -- Accuracy: {best_acc:.2f}%")
        
        # checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'val_history': val_history,
            }, work_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # final results
    results = {
        'best_accuracy': best_acc,
        'final_train_acc': train_history[-1]['acc'],
        'final_val_acc': val_history[-1]['acc'],
        'train_history': train_history,
        'val_history': val_history
    }
    
    with open(work_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {work_dir}")

if __name__ == '__main__':
    main()