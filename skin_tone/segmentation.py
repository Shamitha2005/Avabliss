import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from hrnet import HRNetSegmentation  # Ensure HRNet is correctly implemented
from facenet_pytorch import MTCNN  # Face detection
import pydensecrf.densecrf as dcrf

class HybridFaceSegmentation:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Load DeepLabV3 for coarse segmentation
        self.deeplab = deeplabv3_resnet101(pretrained=True).to(self.device)
        self.deeplab.eval()

        # Load HRNet for fine segmentation (Ensure proper weights are loaded)
        self.hrnet = HRNetSegmentation(pretrained=True).to(self.device)
        self.hrnet.eval()

        # Load MTCNN for accurate face detection
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        # Image preprocessing transformations
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def segment_face(self, image):
        """Segments face using DeepLabV3 and HRNet."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed precision for speedup
                coarse_output = self.deeplab(img_tensor)['out']
                coarse_mask = torch.argmax(coarse_output, dim=1).cpu().numpy()[0]

        # Extract face using MTCNN
        face_region = self.extract_face_region(image_rgb, coarse_mask)

        # Apply super-resolution (Optional, use ESRGAN)
        face_region = self.super_resolve(face_region)

        # Run HRNet on the extracted face
        fine_mask = self.run_hrnet(face_region)

        # Apply CRF for refinement
        fine_mask = self.apply_crf(image_rgb, fine_mask)

        return fine_mask

    def extract_face_region(self, image, mask):
        """Uses MTCNN to extract face accurately, with fallback to the most confident region."""
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None:
            x, y, w, h = map(int, boxes[0])
            return image[y:h, x:w]

        # Fallback: Extract the largest connected component from the coarse mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Ignore background label 0
        largest_mask = (labels == largest_label).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(largest_mask)

        return image[y:y+h, x:x+w]  # Crop the region

    def super_resolve(self, image):
        """Super-resolves an image using ESRGAN (optional)."""
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        return image

    def run_hrnet(self, image):
        """Runs HRNet on the face region for fine segmentation."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fine_output = self.hrnet(img_tensor)
                fine_mask = torch.argmax(fine_output, dim=1).cpu().numpy()[0]

        return fine_mask

    def apply_crf(self, image, mask):
        """Applies Dense CRF for post-processing refinement."""
        h, w = mask.shape
        d = dcrf.DenseCRF2D(w, h, 2)

        # Create unary potential
        unary = np.zeros((2, h, w), dtype=np.float32)
        unary[0] = -np.log(mask + 1e-8)  # Foreground probability
        unary[1] = -np.log(1 - mask + 1e-8)  # Background probability
        unary = unary.reshape(2, -1)

        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=image, compat=10)

        Q = d.inference(5)
        refined_mask = np.argmax(Q, axis=0).reshape(h, w)
        return refined_mask

