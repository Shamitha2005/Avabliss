import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset_loader import FaceSegmentationDataset, load_dataset_paths, split_dataset
from segmentation import HybridFaceSegmentation  
from train import train_model  
from tone import detect_undertone  

# ⚙ Configuration Settings
DATASET_DIRS = [
    r"C:\Users\dell\Desktop\Avabliss\data_sets\CelebAMask-HQ",
    r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD",
    r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD-INDIA",
    r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD-MR",
    r"C:\Users\dell\Desktop\Avabliss\data_sets\fitzpatrick_images"
]
FITZPATRICK_CSV = r"C:\Users\dell\Desktop\Avabliss\data_sets\fitzpatrick17k.csv"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Validate Dataset Paths
print("\n🔍 Checking dataset directories:")
valid_dirs = []
for dir_path in DATASET_DIRS:
    if os.path.exists(dir_path):
        print(f"✅ {dir_path} [FOUND]")
        valid_dirs.append(dir_path)
    else:
        print(f"⚠️ {dir_path} [NOT FOUND]")

if not valid_dirs:
    raise FileNotFoundError("❌ No valid dataset directories found. Please check paths.")

# ✅ Check & Adjust CelebA-HQ Image Directory
celeba_hq_dir = DATASET_DIRS[0]
if os.path.exists(celeba_hq_dir):
    subdirs = os.listdir(celeba_hq_dir)
    print(f"\n📂 Contents of CelebA-HQ: {subdirs}")
    if "images" in subdirs:
        celeba_hq_dir = os.path.join(celeba_hq_dir, "images")
        print("✅ Using 'images' subdirectory for CelebA-HQ dataset.")
DATASET_DIRS[0] = celeba_hq_dir  # Update the main directory

# ✅ Load Image & Mask Paths
print("\n📥 Loading dataset paths...")
image_paths, mask_paths = load_dataset_paths(valid_dirs)
if not image_paths:
    raise ValueError("❌ No images found in dataset directories.")

# ✅ Split Dataset into Train, Val, Test
(train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) = split_dataset(image_paths, mask_paths)

# ✅ Define Dataset & Dataloaders
train_dataset = FaceSegmentationDataset(train_imgs, train_masks, transform=None)
val_dataset = FaceSegmentationDataset(val_imgs, val_masks, transform=None)
test_dataset = FaceSegmentationDataset(test_imgs, test_masks, transform=None)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"\n📊 Dataset Summary:")
print(f"   🟢 Train: {len(train_dataset)} images")
print(f"   🔵 Validation: {len(val_dataset)} images")
print(f"   🟠 Test: {len(test_dataset)} images")

# ✅ Initialize Model & Optimizer
print("\n🧠 Initializing Segmentation Model...")
model = HybridFaceSegmentation().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# ✅ Train Model
print("\n🚀 Starting Training...")
train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE)

# ✅ Evaluation & Skin Tone Detection
def evaluate_model():
    """Evaluates the trained model on a few test images and detects undertones."""
    model.eval()
    with torch.no_grad():
        print("\n🔎 Running Model Evaluation...")
        for img_path in test_imgs[:5]:  # Test on first 5 images
            if not os.path.exists(img_path):
                print(f"⚠️ Skipping missing image: {img_path}")
                continue
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB format

            # 🔥 Run Segmentation Model
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
            segmentation_mask = model(image_tensor)
            segmentation_mask = torch.argmax(segmentation_mask, dim=1).squeeze().cpu().numpy()

            # 🔥 Run Skin Tone Detection
            undertone = detect_undertone(image)

            print(f"🖼 Image: {os.path.basename(img_path)} | 🎨 Detected Undertone: {undertone}")

# ✅ Run Evaluation
evaluate_model()

print("\n✅ All Tasks Completed Successfully!")
