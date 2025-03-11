import os
import zipfile
import pandas as pd
import requests
import cv2
import torch
from torch.utils.data import DataLoader
from dataset_loader import FaceSegmentationDataset, load_dataset_paths, split_dataset
from segmentation import FaceSegmentationModel
from train import train_model
from skin_tone import detect_undertone

# Define dataset paths
ZIP_FILES = [
    "C:/Users/dell/Desktop/Avabliss/data_sets/CelebAMask-HQ.zip",
    "C:/Users/dell/Desktop/Avabliss/data_sets/cfd.zip"
]
EXTRACTED_FOLDER = "C:/Users/dell/Desktop/Avabliss/data_sets/extracted"
FITZPATRICK_CSV = "C:/Users/dell/Desktop/Avabliss/data_sets/fitzpatrick17k (1).csv"
FITZPATRICK_IMAGES_FOLDER = os.path.join(EXTRACTED_FOLDER, "fitzpatrick_images")

# Ensure extraction folder exists
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

# Step 1: Extract ZIP files
for zip_path in ZIP_FILES:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)
    print(f"Extracted: {zip_path}")

# Step 2: Download Fitzpatrick dataset images
os.makedirs(FITZPATRICK_IMAGES_FOLDER, exist_ok=True)

def download_images(csv_path, save_folder):
    df = pd.read_csv(csv_path)
    url_column = "url"  # Adjust column name if different

    for idx, url in enumerate(df[url_column]):
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                image_path = os.path.join(save_folder, f"fitzpatrick_{idx}.jpg")
                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded: {image_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

download_images(FITZPATRICK_CSV, FITZPATRICK_IMAGES_FOLDER)

# Step 3: Load all datasets
DATASET_DIRS = [EXTRACTED_FOLDER, FITZPATRICK_IMAGES_FOLDER]

# Training configurations
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset paths
image_paths, mask_paths = load_dataset_paths(DATASET_DIRS)
(train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) = split_dataset(image_paths, mask_paths)

# Define dataset and dataloaders
train_dataset = FaceSegmentationDataset(train_imgs, train_masks, transform=None)
val_dataset = FaceSegmentationDataset(val_imgs, val_masks, transform=None)
test_dataset = FaceSegmentationDataset(test_imgs, test_masks, transform=None)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize segmentation model
model = FaceSegmentationModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Train the segmentation model
train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)

# Step 4: Evaluate on test images and apply undertone detection
def evaluate_model():
    model.eval()
    with torch.no_grad():
        for img_path in test_imgs[:5]:  # Test on first 5 images
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            undertone = detect_undertone(image)
            print(f"Image: {img_path}, Detected Undertone: {undertone}")

# Run evaluation
evaluate_model()
