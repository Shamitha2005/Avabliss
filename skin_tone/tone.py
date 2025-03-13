import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from sklearn.cluster import KMeans
from torchvision import models

class SkinUndertoneClassifier(nn.Module):
    def __init__(self, num_classes=30):  # Further expanded undertones
        super(SkinUndertoneClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def preprocess_image(image):
    """Convert to CIELAB, extract skin regions, and normalize."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return image

def extract_skin_regions(image):
    """Apply a clustering-based skin extraction method."""
    image_lab = preprocess_image(image)
    img_flattened = image_lab.reshape((-1, 3))
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(img_flattened)
    mask = (kmeans.labels_.reshape(image.shape[:2]) == 1).astype(np.uint8)
    skin_pixels = image_lab[mask == 1]
    return skin_pixels

def calculate_hue_angle_luminance(skin_pixels):
    """Compute hue angle and luminance from skin pixels in CIELAB space."""
    L, a, b = np.mean(skin_pixels, axis=0)
    hue_angle = np.arctan2(b, a) * (180 / np.pi)
    return hue_angle, L

def classify_undertone(hue_angle, L):
    """Expanded classification into detailed undertones, considering luminance."""
    undertones = [
        "Pale Neutral", "Ivory", "Light Beige", "Golden Beige", "Peach", "Warm Yellow", "Olive", "Muted Olive", "Green Undertone", "Blue Undertone",
        "Cool Pink", "Cool Beige", "Neutral", "Warm Red", "Golden Yellow", "Rich Tan", "Deep Golden", "Deep Warm Brown", "Deep Olive", "Deep Neutral",
        "Rich Ebony", "Dark Cool Brown", "Reddish Brown", "Deep Red", "Cool Deep Brown", "Mahogany", "Ebony", "Deep Blue Undertone", "Muted Cool", "Ultra Deep Neutral"
    ]
    
    index = int(((hue_angle + 180) / 360) * len(undertones)) % len(undertones)
    if L < 50:
        return undertones[index] + " (Deep)"
    else:
        return undertones[index]

def detect_undertone(image):
    """Full pipeline: Extract skin, compute hue angle, and classify undertone."""
    skin_pixels = extract_skin_regions(image)
    if len(skin_pixels) == 0:
        return "Undetermined"
    hue_angle, L = calculate_hue_angle_luminance(skin_pixels)
    undertone = classify_undertone(hue_angle, L)
    return undertone

# Example Usage:
# image = cv2.imread("face.jpg")
# undertone = detect_undertone(image)
# print("Detected Undertone:", undertone)
print("done!!")