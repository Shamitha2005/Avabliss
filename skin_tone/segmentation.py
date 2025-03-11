import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from facenet_pytorch import MTCNN  # Face detection
import pydensecrf.densecrf as dcrf

class HybridFaceSegmentation(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(HybridFaceSegmentation, self).__init__()
        self.device = device

        # Load DeepLabV3+ for segmentation
        self.deeplab = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(self.device)
        self.deeplab.eval()

        # Load MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        # Image preprocessing transformations
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def optimize_lighting(self, image):
        """Applies lighting correction techniques: Gamma, CLAHE, and white balance."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Gamma Correction
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)

        # Adaptive Histogram Equalization (CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # Convert back to RGB

        return image

    def forward(self, image):
        """Segments face using DeepLabV3+ with lighting optimization."""
        image = self.optimize_lighting(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB consistency
        img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                segmentation_output = self.deeplab(img_tensor)['out']
                segmentation_mask = torch.argmax(segmentation_output, dim=1).cpu().numpy()[0]

        refined_mask = self.apply_crf(image_rgb, segmentation_mask)
        return refined_mask

    def apply_crf(self, image, mask):
        """Applies Dense CRF for post-processing refinement."""
        h, w = mask.shape
        d = dcrf.DenseCRF2D(w, h, 2)

        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0

        unary = np.zeros((2, h, w), dtype=np.float32)
        unary[0] = -np.log(1 - mask + 1e-8)  # Foreground
        unary[1] = -np.log(mask + 1e-8)      # Background
        unary = unary.reshape(2, -1)

        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=image, compat=10)

        Q = d.inference(5)
        refined_mask = np.argmax(Q, axis=0).reshape(h, w)
        return refined_mask
