"""
Kiến trúc mạng học sâu CNN (MobileNetV3-Small Transfer Learning)
dùng chung cho quá trình huấn luyện (training/train_cnn.py) và suy luận thời gian thực (backend/ai_engine.py).
"""

from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms

# Lưu bộ nhớ đệm trọng số ImageNet ngay trong thư mục models/torch_cache của dự án
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent.parent / "models" / "torch_cache"))

# Thứ tự 4 lớp chuẩn theo bảng chữ cái ASCII (khớp tuyệt đối với torchvision.datasets.ImageFolder)
CLASS_NAMES: List[str] = ["Closed", "Open", "no_yawn", "yawn"]

CLASS_LABELS_VI: Dict[str, str] = {
    "Closed": "Nhắm mắt (Closed)",
    "Open": "Mở mắt (Open)",
    "no_yawn": "Bình thường (No Yawn)",
    "yawn": "Ngáp (Yawn)",
}

IMG_SIZE: int = 128
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class DriverStateCNN(nn.Module):
    """
    Mạng CNN phân loại 4 trạng thái mắt & miệng tài xế dựa trên MobileNetV3-Small.
    Hỗ trợ đồng thời ảnh màu RGB ban ngày (Yawn-Eye Dataset) và ảnh hồng ngoại ban đêm (MRL Eye Dataset).
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Chuẩn hóa ảnh đầu vào cho tập Validation / Test / Suy luận thời gian thực."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Data Augmentation mô phỏng điều kiện ánh sáng ngày/đêm, hồng ngoại và góc nghiêng đầu."""
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.12), int(img_size * 1.12))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.04),
        transforms.RandomGrayscale(p=0.15),  # Tăng cường tương thích camera hồng ngoại ban đêm
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_trained_cnn(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[DriverStateCNN, List[str], int, dict]:
    """Tải mô hình DriverStateCNN từ file trọng số .pth."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        class_names = ckpt.get("class_names", CLASS_NAMES)
        img_size = int(ckpt.get("img_size", IMG_SIZE))
        model = DriverStateCNN(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        class_names = CLASS_NAMES
        img_size = IMG_SIZE
        model = DriverStateCNN(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(ckpt)
        meta = {}

    model.to(device)
    model.eval()
    return model, class_names, img_size, meta
