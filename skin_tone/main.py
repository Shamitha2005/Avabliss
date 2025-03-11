import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset_loader import FaceSegmentationDataset, load_dataset_paths, split_dataset
from segmentation import HybridFaceSegmentation  # ✅ Updated import
from train import train_model
from skin_tone import detect_undertone

# Configuration settings
DATASET_DIRS = [r"C:\Users\dell\Desktop\Avabliss\data_sets\CelebAMask-HQ.zip",
                r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd.zip"]
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset paths
image_paths, mask_paths = load_dataset_paths(DATASET_DIRS)
(train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) = split_dataset(image_paths, mask_paths)

# Define dataset loaders
train_dataset = FaceSegmentationDataset(train_imgs, train_masks, transform=None)
val_dataset = FaceSegmentationDataset(val_imgs, val_masks, transform=None)
test_dataset = FaceSegmentationDataset(test_imgs, test_masks, transform=None)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize Hybrid Segmentation Model
model = HybridFaceSegmentation(device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Train the segmentation model
train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)

# Evaluation & Skin Tone Detection
def evaluate_model():
    model.eval()
    with torch.no_grad():
        for img_path in test_imgs[:5]:  # Test on first 5 images
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
            segmentation_mask = model(image)  # 🔥 Run segmentation model
            undertone = detect_undertone(image)  # 🔥 Run undertone detection

            print(f"Image: {img_path}, Detected Undertone: {undertone}")

# Run Evaluation
evaluate_model()
