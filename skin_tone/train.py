import os
import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import sys

# Add the current directory to Python’s path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dataset_loader import FaceSegmentationDataset
from segmentation import HybridFaceSegmentation

import torch.nn as nn
import torch.nn.functional as F

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth.tar"):
    """Saves model checkpoint."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

class HybridLoss(nn.Module):
    """Combines Dice Loss, Focal Loss, and Cross-Entropy Loss."""
    def __init__(self, alpha=0.5, gamma=2, ce_weight=None):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)

    def dice_loss(self, preds, targets, smooth=1):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum(dim=(2, 3))
        return 1 - ((2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)).mean()

    def focal_loss(self, preds, targets):
        ce_loss = self.ce_loss(preds, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

    def forward(self, preds, targets):
        return self.dice_loss(preds, targets) + self.focal_loss(preds, targets) + self.ce_loss(preds, targets)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()  # Use AMP only if CUDA is available
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

# Data Augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # Removed alpha_affine
    A.CLAHE(p=0.2),
    A.RandomGamma(p=0.2),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# Function to set dataset paths dynamically
def set_datasets(image_paths, mask_paths):
    global train_loader
    if not image_paths or not mask_paths:
        raise ValueError("Image paths and mask paths must be provided before training.")
    
    train_dataset = FaceSegmentationDataset(image_paths, mask_paths, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Dataset loaded: {len(train_dataset)} samples")

# Model, Loss, Optimizer
model = HybridFaceSegmentation().to(DEVICE)
criterion = HybridLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

<<<<<<< HEAD
# Training Loop
def train():
    if 'train_loader' not in globals() or train_loader is None:
        raise ValueError("Dataset not loaded! Call set_datasets(image_paths, mask_paths) before training.")
    
=======
# Training Function: Now accepts parameters
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """Trains the model on the dataset."""
>>>>>>> 73fc4f363685c0238103529b2a8f2a5298ee7136
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
        save_checkpoint(model, optimizer, epoch)

if __name__ == "__main__":
    print("Train script loaded. Please call set_datasets(image_paths, mask_paths) from main.py before training.")
