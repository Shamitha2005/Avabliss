import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_loader import FaceSegmentationDataset, load_dataset_paths
from segmentation import HybridFaceSegmentation
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

def split_dataset(image_paths, mask_paths, test_size=0.1, val_size=0.1):
    """Splits dataset into train, validation, and test sets."""
    valid_indices = [i for i, m in enumerate(mask_paths) if m is not None]
    valid_imgs = [image_paths[i] for i in valid_indices]
    valid_masks = [mask_paths[i] for i in valid_indices]

    train_img, test_img, train_mask, test_mask = train_test_split(
        valid_imgs, valid_masks, test_size=test_size, random_state=42
    )
    train_img, val_img, train_mask, val_mask = train_test_split(
        train_img, train_mask, test_size=val_size, random_state=42
    )

    return (train_img, train_mask), (val_img, val_mask), (test_img, test_mask)

def prepare_dataloaders(image_paths, mask_paths):
    """Handles dataset splitting and dataloading."""
    (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) = split_dataset(image_paths, mask_paths)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.CLAHE(p=0.2),
        A.RandomGamma(p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    train_dataset = FaceSegmentationDataset(train_imgs, train_masks, transform=transform)
    val_dataset = FaceSegmentationDataset(val_imgs, val_masks, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)

    return train_loader, val_loader

# Hybrid Loss Function
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

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth.tar"):
    """Saves model checkpoint."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

# Training Function
def train_model(image_paths, mask_paths):
    """Trains the segmentation model."""
    train_loader, val_loader = prepare_dataloaders(image_paths, mask_paths)

<<<<<<< HEAD
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
=======
    model = HybridFaceSegmentation().to(DEVICE)
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

>>>>>>> 116ffa08ad982f2e326d74fe1c9cf936deab2160
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")
        save_checkpoint(model, optimizer, epoch)

    return model, val_loader

if __name__ == "__main__":
    print("Train script loaded. Call train_model(image_paths, mask_paths) from main.py to start training.")
