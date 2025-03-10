import os

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
import torchvision.transforms as T

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
        img_dir = os.path.join(dataset_dir, "images")
        mask_dir = os.path.join(dataset_dir, "masks")

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda f: process_file(f, img_dir, mask_dir), os.listdir(img_dir)))

        for img_path, mask_path in results:
            if os.path.exists(img_path):  # Ensuring valid paths
                image_paths.append(img_path)
                mask_paths.append(mask_path if mask_path and os.path.exists(mask_path) else None)

    return image_paths, mask_paths

def preprocess_image(image):
    """Applies OpenCV preprocessing steps and converts to CIELAB color space."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Noise reduction
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Convert to CIELAB for skin-tone processing
    return image

def quick_mask(image):
    """Lightweight mask generation using Otsu's thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_LAB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)

def map_mask(mask, mapping):
    """Dynamically standardize mask labels using a provided mapping."""
    return np.vectorize(lambda x: mapping.get(x, 0))(mask).astype(np.float32)

def split_dataset(image_paths, mask_paths, test_size=0.1, val_size=0.1):
    """Stratified split of dataset into train, validation, and test sets, handling missing masks."""
    valid_indices = [i for i, m in enumerate(mask_paths) if m is not None]
    valid_imgs = [image_paths[i] for i in valid_indices]
    valid_masks = [mask_paths[i] for i in valid_indices]

    train_img, test_img, train_mask, test_mask = train_test_split(
        valid_imgs, valid_masks, test_size=test_size, stratify=valid_masks, random_state=42
    )
    train_img, val_img, train_mask, val_mask = train_test_split(
        train_img, train_mask, test_size=val_size, stratify=valid_masks[:len(train_mask)], random_state=42
    )

    return (train_img, train_mask), (val_img, val_mask), (test_img, test_mask)

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
        image = preprocess_image(image)

        if self.mask_paths[idx]:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = map_mask(mask, self.dataset_mappings.get(self.dataset_name, {}))
        else:
            mask = quick_mask(image)  # Use lightweight Otsu thresholding

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.CLAHE(p=0.2),
            A.RandomGamma(p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        transformed = transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        if self.use_fp16:
            image = image.half()  # Convert image to FP16 for performance optimization

        return image, mask
print("done")