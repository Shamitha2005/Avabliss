import os
import torch
import cv2
import numpy as np
from dataset_loader import load_dataset_paths
from train import train_model, prepare_dataloaders, split_dataset
from segmentation import HybridFaceSegmentation
from tone import detect_undertone  

def main():
    # ⚙ Configuration Settings
    DATASET_DIRS = [
        r"C:\Users\dell\Desktop\Avabliss\data_sets\CelebAMask-HQ",
        r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD",
        r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD-INDIA",
        r"C:\Users\dell\Desktop\Avabliss\data_sets\cfd\CFD Version 3.0\Images\CFD-MR",
        r"C:\Users\dell\Desktop\Avabliss\data_sets\fitzpatrick_images"
    ]
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ✅ Validate Dataset Paths
    print("\n🔍 Checking dataset directories...")
    valid_dirs = [d for d in DATASET_DIRS if os.path.exists(d)]
    if not valid_dirs:
        raise FileNotFoundError("❌ No valid dataset directories found. Please check paths.")
    
    for d in valid_dirs:
        print(f"✅ {d} [FOUND]")
    
    # ✅ Load Image & Mask Paths
    print("\n📥 Loading dataset paths...")
    image_paths, mask_paths = load_dataset_paths(valid_dirs)
    if not image_paths:
        raise ValueError("❌ No images found in dataset directories.")
    
    # ✅ Split the Dataset for Evaluation Purposes
    # We use the same split parameters as in train.py (random_state is fixed for reproducibility)
    _, _, (test_imgs, test_masks) = split_dataset(image_paths, mask_paths)
    
    # ✅ Prepare Data Loaders for Training & Validation
    train_loader, val_loader = prepare_dataloaders(image_paths, mask_paths)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   🟢 Train: {len(train_loader.dataset)} images")
    print(f"   🔵 Validation: {len(val_loader.dataset)} images")
    
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
            # Evaluate on the first 5 test images
            for img_path in test_imgs[:5]:
                if not os.path.exists(img_path):
                    print(f"⚠️ Skipping missing image: {img_path}")
                    continue
                
                image = cv2.imread(img_path)
                # Ensure the image is in RGB format for both segmentation and undertone detection
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
                # 🔥 Run Segmentation Model
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
                segmentation_output = model(image_tensor)
                segmentation_mask = torch.argmax(segmentation_output, dim=1).squeeze().cpu().numpy()
    
                # 🔥 Run Skin Tone Detection
                undertone = detect_undertone(image)
    
                print(f"🖼 Image: {os.path.basename(img_path)} | 🎨 Detected Undertone: {undertone}")
    
    # Run Evaluation
    evaluate_model()
    
    print("\n✅ All Tasks Completed Successfully!")

if __name__ == '__main__':
    main()
 