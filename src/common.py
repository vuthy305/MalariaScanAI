"""Shared training and evaluation code used by notebooks 02-05.

Kept in one file so the four experiments are trained and scored in exactly
the same way (fair comparison) and the notebooks stay short.
"""
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.model import IMG_SIZE, MEAN, STD, EVAL_TRANSFORM, build_model, load_model  # noqa: E402

DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"            # mixed precision only on GPU
WORKERS = 0 if os.name == "nt" else 2      # Windows notebooks need 0 workers

# The four training runs of the project.
#   finetune=False -> basic training: all layers trained together, fixed learning rate
#   finetune=True  -> two-stage fine-tuning: (1) train only the new head,
#                     (2) unfreeze layer3 + layer4 and train with a cosine LR schedule
EXPERIMENTS = {
    "resnet18":          dict(arch="resnet18", augment=False, finetune=False),  # Baseline 1
    "resnet50":          dict(arch="resnet50", augment=False, finetune=False),  # Baseline 2 = Ablation A
    "resnet50_aug":      dict(arch="resnet50", augment=True,  finetune=False),  # Ablation B
    "proposed_resnet50": dict(arch="resnet50", augment=True,  finetune=True),   # Proposed = Ablation C
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_environment():
    print("Python :", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU name:", props.name)
        print(f"GPU memory: {props.total_memory / 1024 ** 3:.1f} GB")
    if (DATA / "train").exists():
        loaders, classes = get_loaders(augment=False)
        for split, loader in loaders.items():
            print(f"{split} images: {len(loader.dataset)}")
        print("Number of classes:", len(classes))
        print("Classes:", classes)


def get_train_transform(augment):
    if not augment:
        return EVAL_TRANSFORM
    # A blood cell has no "right way up", so flips and any rotation keep the
    # label correct (the corners are already black in this dataset).
    # Colour changes are kept small because the parasite shows up as a stain
    # colour, and no cropping is used so a parasite near the edge is never cut out.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_loaders(augment, batch_size=16):
    """Validation and test never use random augmentation."""
    sets = {"train": datasets.ImageFolder(DATA / "train", get_train_transform(augment)),
            "val": datasets.ImageFolder(DATA / "val", EVAL_TRANSFORM),
            "test": datasets.ImageFolder(DATA / "test", EVAL_TRANSFORM)}
    for split in ("val", "test"):
        assert sets[split].classes == sets["train"].classes, f"class mismatch in {split}"
    loaders = {s: DataLoader(d, batch_size, shuffle=(s == "train"), num_workers=WORKERS,
                             pin_memory=USE_AMP) for s, d in sets.items()}
    return loaders, sets["train"].classes


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def print_parameters(title, model):
    c = count_parameters(model)
    print(f"{title}\n"
          f"Total parameters:         {c['total']:,}\n"
          f"Trainable parameters:     {c['trainable']:,}\n"
          f"Non-trainable parameters: {c['non_trainable']:,}\n")
    return c


def set_trainable(model, prefixes=None):
    """Train only layers whose name starts with one of `prefixes` (None = all)."""
    for name, p in model.named_parameters():
        p.requires_grad = prefixes is None or name.startswith(prefixes)


def compute_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                  zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": p, "recall": r, "macro_f1": f1}


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    """One pass over `loader`. Trains if an optimizer is given, else only evaluates."""
    training = optimizer is not None
    model.train(training)
    loss_sum, y_true, y_pred = 0.0, [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(training), \
                torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            out = model(x)
            loss = criterion(out, y)
        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * len(y)
        y_true += y.tolist()
        y_pred += out.argmax(1).tolist()
    return {"loss": loss_sum / len(y_true), **compute_metrics(y_true, y_pred)}, y_true, y_pred


def train(name, epochs=10, patience=3, batch_size=16, lr=1e-4, head_epochs=2,
          pretrained=True, seed=42):
    """Train one experiment. Saves models/<name>_best.pth (best validation
    Macro-F1) and results/<name>_history.json."""
    cfg = EXPERIMENTS[name]
    set_seed(seed)
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    loaders, classes = get_loaders(cfg["augment"], batch_size)
    class_to_idx = loaders["train"].dataset.class_to_idx
    (MODELS / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2))

    model = build_model(cfg["arch"], len(classes), pretrained).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except AttributeError:  # older PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    def new_optimizer(learning_rate):
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=learning_rate, weight_decay=1e-4)

    scheduler = None
    if cfg["finetune"]:
        set_trainable(model, ("fc",))                      # stage 1: head only
        print_parameters(f"{name}, stage 1 (head only, {head_epochs} epochs)", model)
        optimizer = new_optimizer(1e-3)
    else:
        params = print_parameters(name, model)             # all layers trainable
        optimizer = new_optimizer(lr)

    history, best_f1, bad_epochs, start = [], -1.0, 0, time.time()
    for epoch in range(1, epochs + 1):
        if cfg["finetune"] and epoch == head_epochs + 1:   # stage 2: fine-tune
            set_trainable(model, ("layer3", "layer4", "fc"))
            params = print_parameters(f"{name}, stage 2 (fine-tune layer3 + layer4 + head)",
                                      model)
            optimizer = new_optimizer(lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - head_epochs)

        t0 = time.time()
        tr, _, _ = run_epoch(model, loaders["train"], criterion, optimizer, scaler)
        va, _, _ = run_epoch(model, loaders["val"], criterion)
        if scheduler:
            scheduler.step()
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        print(f"Epoch {epoch}/{epochs}\n"
              f"Train Loss: {tr['loss']:.3f}\n"
              f"Train Accuracy: {tr['accuracy'] * 100:.2f}%\n"
              f"Val Loss: {va['loss']:.3f}\n"
              f"Val Accuracy: {va['accuracy'] * 100:.2f}%\n"
              f"Val Macro-F1: {va['macro_f1'] * 100:.2f}%\n"
              f"Time: {time.time() - t0:.1f} sec\n")

        if va["macro_f1"] > best_f1:                       # best checkpoint
            best_f1, bad_epochs = va["macro_f1"], 0
            torch.save({"arch": cfg["arch"], "state_dict": model.state_dict()},
                       MODELS / f"{name}_best.pth")
        elif not cfg["finetune"] or epoch > head_epochs:   # early stopping
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break

    info = {"name": name, **cfg, "params": params, "best_val_macro_f1": best_f1,
            "train_time_sec": time.time() - start, "history": history}
    (RESULTS / f"{name}_history.json").write_text(json.dumps(info, indent=2))
    print(f"Best Val Macro-F1: {best_f1 * 100:.2f}%   "
          f"Training time: {info['train_time_sec']:.0f} sec")
    print(f"Saved models/{name}_best.pth, models/class_to_idx.json, "
          f"results/{name}_history.json")
    return info


def evaluate(name, split="test"):
    """Score a saved checkpoint.
    Returns (metrics dict, confusion matrix, class names, classification report)."""
    model, classes = load_model(MODELS / f"{name}_best.pth",
                                MODELS / "class_to_idx.json", DEVICE)
    loaders, _ = get_loaders(augment=False)
    m, y_true, y_pred = run_epoch(model, loaders[split], nn.CrossEntropyLoss())
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   zero_division=0)
    return m, cm, classes, report


def plot_history(name, save_path=None):
    """Loss, accuracy and Macro-F1 curves for one run."""
    import matplotlib.pyplot as plt
    h = json.loads((RESULTS / f"{name}_history.json").read_text())["history"]
    epochs = [e["epoch"] for e in h]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, title in zip(axes, ("loss", "accuracy", "macro_f1"),
                              ("Loss", "Accuracy", "Macro-F1")):
        ax.plot(epochs, [e[f"train_{key}"] for e in h], marker="o", label="Training")
        ax.plot(epochs, [e[f"val_{key}"] for e in h], marker="o", label="Validation")
        ax.set_title(f"{name}: Training vs Validation {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
