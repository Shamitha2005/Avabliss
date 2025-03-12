import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from facenet_pytorch import MTCNN  

class HybridFaceSegmentation(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(HybridFaceSegmentation, self).__init__()
        self.device = device
        self.deeplab = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(self.device)
        self.deeplab.classifier[4] = nn.Conv2d(256, 5, kernel_size=(1,1))  # Classes: skin, eyes, lips, blush, background
        self.deeplab.eval()
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def forward(self, image):
        """Detects and segments facial parts."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.deeplab(img_tensor)['out']
            segmentation_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
        return segmentation_mask
