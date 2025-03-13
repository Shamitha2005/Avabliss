import torch
import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from dataset_loader import FaceSegmentationDataset, load_dataset_paths, split_dataset
from segmentation import HybridFaceSegmentation  # ✅ Updated import
from train import train_model
from tone import detect_undertone

# Configuration settings
DATASET_DIRS = [r"C:\Users\dell\Desktop\Avabliss\data_sets\CelebAMask-HQ\CelebA-HQ-img",
                r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD",
                r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD-INDIA",
                r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD-MR"]
FITZPATRICK_CSV = r"C:\Users\dell\Desktop\Avabliss\data_sets\fitzpatrick17k.csv"
FITZPATRICK_DIR = r"C:\Users\dell\Desktop\Avabliss\data_sets\fitzpatrick_images"
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure Fitzpatrick CSV exists
if not os.path.exists(FITZPATRICK_CSV):
    raise FileNotFoundError(f"Error: Fitzpatrick CSV not found at {FITZPATRICK_CSV}")

# Debugging: Print dataset directories
print("Checking dataset directories:")
for dir_path in DATASET_DIRS:
    print(f" - {dir_path} -> Exists: {os.path.exists(dir_path)}")
    if not os.path.exists(dir_path):
        print(f"Warning: Directory not found - {dir_path}")

# Check and adjust CelebA-HQ image directory
celeba_hq_dir = DATASET_DIRS[0]
if os.path.exists(celeba_hq_dir):
    subdirs = os.listdir(celeba_hq_dir)
    print(f"Contents of CelebA-HQ-img: {subdirs}")
    image_files = [f for f in subdirs if f.endswith(('.jpg', '.png'))]
    if image_files:
        print("✅ Found images directly inside CelebA-HQ-img. Using this directory.")
    elif "images" in subdirs:
        print("✅ Found 'images' subdirectory. Updating path.")
        celeba_hq_dir = os.path.join(celeba_hq_dir, "images")
    else:
        raise FileNotFoundError("❌ No images found in CelebA-HQ-img. Check dataset structure.")
    DATASET_DIRS[0] = celeba_hq_dir

# Load dataset paths
image_paths, mask_paths = load_dataset_paths([d for d in DATASET_DIRS if os.path.exists(d)])
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
            if not os.path.exists(img_path):
                print(f"Skipping missing image: {img_path}")
                continue
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format
            segmentation_mask = model(image)  # 🔥 Run segmentation model
            undertone = detect_undertone(image)  # 🔥 Run undertone detection

            print(f"Image: {img_path}, Detected Undertone: {undertone}")

# Run Evaluation
evaluate_model()
