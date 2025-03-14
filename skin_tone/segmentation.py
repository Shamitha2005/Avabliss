import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from facenet_pytorch import MTCNN  

class HybridFaceSegmentation(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(HybridFaceSegmentation, self).__init__()
        self.device = device
        self.deeplab = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(self.device)
        # Modify classifier for 5 classes: skin, eyes, lips, blush, background
        self.deeplab.classifier[4] = nn.Conv2d(256, 5, kernel_size=(1,1))
        self.deeplab.eval()
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        # Transformation to apply for raw NumPy images (assumed HWC, RGB)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def forward(self, image):
        """
        If image is a torch.Tensor, assume it's already transformed.
        Otherwise, if image is a NumPy array (3D: HWC), apply self.transform.
        """
        if isinstance(image, torch.Tensor):
            # Assume the image is already a tensor with shape [B, C, H, W] or [C, H, W]
            # If not batched (i.e. 3D), add batch dimension
            if image.ndim == 3:
                img_tensor = image.unsqueeze(0).to(self.device)
            else:
                img_tensor = image.to(self.device)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:  # HWC
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            elif image.ndim == 4:  # Already batched as numpy array
                img_tensor = torch.from_numpy(image).to(self.device)
            else:
                raise ValueError("Input numpy array must be 3 or 4 dimensions.")
        else:
            raise TypeError("Unsupported input type for segmentation model forward pass.")

        with torch.no_grad():
            output = self.deeplab(img_tensor)['out']
            segmentation_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
        return segmentation_mask
