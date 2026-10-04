"""Model definition shared by training (src/common.py) and the web backend."""
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics

# Deterministic preprocessing. Used for validation, test AND the web app,
# so inference always matches evaluation.
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model(arch, num_classes, pretrained=False):
    """ResNet-18 or ResNet-50 with a new classification head."""
    model = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path, mapping_path, device):
    """Load a checkpoint and its class mapping. Returns (model, class names)."""
    class_to_idx = json.loads(Path(mapping_path).read_text())
    classes = sorted(class_to_idx, key=class_to_idx.get)  # index order
    ckpt = torch.load(model_path, map_location=device)
    model = build_model(ckpt["arch"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), classes
