"""predict_image(): image in, predicted class + confidence + ranked classes out.

With explain=True it also returns a Grad-CAM heatmap: a picture showing which
part of the image influenced the model's answer the most.
"""
import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.model import EVAL_TRANSFORM, IMG_SIZE, load_model

MODELS = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS / "best_model.pth"
MAPPING_PATH = MODELS / "class_to_idx.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model, _classes = None, None


def get_model():
    """Load the model once and reuse it for every request."""
    global _model, _classes
    if _model is None:
        _model, _classes = load_model(MODEL_PATH, MAPPING_PATH, DEVICE)
    return _model, _classes


def _grad_cam(rgb, activations, score):
    """Grad-CAM (Selvaraju et al., 2017) on the last convolutional block.

    activations: output of model.layer4 for this image, shape (1, C, 7, 7)
    score:       the model's raw output for the predicted class
    Returns a PNG (as a data URI) of the image with the heatmap drawn on top.
    """
    grads = torch.autograd.grad(score, activations)[0]        # how much each feature map matters
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # scale to 0..1

    # Simple blue -> green -> red colour scale (no extra libraries needed).
    heat = np.stack([np.clip(1.5 - np.abs(4 * cam - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * cam - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)], axis=-1)
    base = np.asarray(rgb.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255
    alpha = (0.6 * cam)[..., None]                            # stronger colour where it matters more
    overlay = np.uint8(255 * ((1 - alpha) * base + alpha * heat))

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def predict_image(image, top_k=3, explain=False):
    """`image` is a file path or a PIL image."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    rgb = image.convert("RGB")
    model, classes = get_model()
    x = EVAL_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)

    kept = {}
    hook = model.layer4.register_forward_hook(lambda module, inp, out: kept.update(act=out))
    with torch.set_grad_enabled(explain):                     # gradients only when explaining
        logits = model(x)
    hook.remove()

    probs = torch.softmax(logits.detach(), dim=1)[0]
    conf, idx = probs.topk(min(top_k, len(classes)))
    top3 = [{"label": classes[i], "confidence": round(c, 4)}
            for c, i in zip(conf.tolist(), idx.tolist())]
    result = {"prediction": top3[0]["label"],
              "confidence": top3[0]["confidence"],
              "top3": top3}
    if explain:
        result["heatmap"] = _grad_cam(rgb, kept["act"], logits[0, idx[0]])
    return result