import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

def load_dataset_paths(dataset_dirs):
    """Loads image and mask paths from multiple datasets, handling different formats and missing masks."""
    image_paths = []
    mask_paths = []
    mask_extensions = [".png", ".bmp", ".tiff", ".jpg"]

    def process_file(filename, img_dir, mask_dir):
        img_path = os.path.join(img_dir, filename)
        mask_path = None
        for ext in mask_extensions:
            potential_mask_path = os.path.join(mask_dir, filename.rsplit(".", 1)[0] + ext)
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break
        return img_path, mask_path

    for dataset_dir in dataset_dirs:
        img_dir = os.path.join(dataset_dir, "images") if os.path.exists(os.path.join(dataset_dir, "images")) else dataset_dir
        mask_dir = os.path.join(dataset_dir, "masks") if os.path.exists(os.path.join(dataset_dir, "masks")) else dataset_dir

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda f: process_file(f, img_dir, mask_dir), os.listdir(img_dir)))

        for img_path, mask_path in results:
            if os.path.exists(img_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path if mask_path and os.path.exists(mask_path) else None)

    return image_paths, mask_paths


class FaceSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, dataset_name="Generic", use_fp16=False, dataset_mappings=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.dataset_name = dataset_name
        self.use_fp16 = use_fp16
        self.dataset_mappings = dataset_mappings if dataset_mappings else {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_paths[idx]:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Empty mask for missing labels

        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        transformed = transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        if self.use_fp16:
            image = image.half()

        return image, mask
"NOthing"