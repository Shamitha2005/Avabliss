import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import sys

sys.path.append(r"C:\Users\dell\Desktop\Avabliss\HRNet-Semantic-Segmentation\lib")

from models.seg_hrnet import HighResolutionNet
from facenet_pytorch import MTCNN
import pydensecrf.densecrf as dcrf

# HRNet Configuration
HRNET_CONFIG = {
    "MODEL": {
        "ALIGN_CORNERS": True,
        "EXTRA": {
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [1]},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2]},
            "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2]},
            "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2, 2]},
        },
    }
}

class HybridFaceSegmentation:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.deeplab = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(self.device)
        self.deeplab.eval()
        self.hrnet = HighResolutionNet(HRNET_CONFIG).to(self.device)
        self.hrnet.eval()
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def optimize_lighting(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image

    def segment_face(self, image):
        image = self.optimize_lighting(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                coarse_output = self.deeplab(img_tensor)['out']
                coarse_mask = torch.argmax(coarse_output, dim=1).cpu().numpy()[0]

        face_region = self.extract_face_region(image_rgb, coarse_mask)
        face_region = self.super_resolve(face_region)
        fine_mask = self.run_hrnet(face_region)
        fine_mask = self.apply_crf(image_rgb, fine_mask)
        return fine_mask

    def extract_face_region(self, image, mask):
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            x, y, w, h = map(int, boxes[0])
            return image[y:h, x:w]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = (labels == largest_label).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(largest_mask)
        return image[y:y+h, x:x+w]

    def super_resolve(self, image):
        return cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)

    def run_hrnet(self, image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fine_output = self.hrnet(img_tensor)
                fine_mask = torch.argmax(fine_output, dim=1).cpu().numpy()[0]
        return fine_mask

    def apply_crf(self, image, mask):
        h, w = mask.shape
        d = dcrf.DenseCRF2D(w, h, 2)
        mask = (mask > 0.5).astype(np.float32)
        unary = np.zeros((2, h, w), dtype=np.float32)
        unary[0] = -np.log(mask + 1e-8)
        unary[1] = -np.log(1 - mask + 1e-8)
        unary = unary.reshape(2, -1)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=image, compat=10)
        Q = d.inference(5)
        refined_mask = np.argmax(Q, axis=0).reshape(h, w)
        return refined_mask
