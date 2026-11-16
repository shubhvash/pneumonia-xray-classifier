"""Shared code for the notebook and train.py: data loading, patient-level
splitting, preprocessing, model construction and the training loop.

The preprocessing here (resize -> 3-channel -> ImageNet normalisation) is
re-implemented with plain PIL + NumPy in predict.py so the web service does
not need PyTorch. Keep the two in sync.
"""
from __future__ import annotations

import copy
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DATA_DIR = Path(__file__).parent / "data" / "chest_xray"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index == label
SEED = 42


# --------------------------------------------------------------------------
# Reproducibility / device
# --------------------------------------------------------------------------
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def free_memory() -> None:
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Data catalogue + patient-level split
# --------------------------------------------------------------------------
def patient_id(filename: str) -> str:
    """Derive a patient identifier from the file name.

    INFECTED: person100_bacteria_475.jpeg  -> person100
    NORMAL:   IM-0031-0001.jpeg            -> IM-0031
              NORMAL2-IM-0272-0001.jpeg    -> NORMAL2-IM-0272
    """
    m = re.match(r"(person\d+)_", filename)
    if m:
        return m.group(1)
    m = re.match(r"((?:NORMAL2-)?IM-\d+)", filename)
    if m:
        return m.group(1)
    return Path(filename).stem


def subtype(folder: str, filename: str) -> str:
    if folder == "NORMAL":
        return "normal"
    return "virus" if "virus" in filename else "bacteria"


def build_catalogue(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("*/*.jp*g")):
        folder = path.parent.name
        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "folder": folder,
                "label": int(folder == "INFECTED"),
                "subtype": subtype(folder, path.name),
                "patient": patient_id(path.name),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No images found under {data_dir}. See README -> Dataset.")
    return pd.DataFrame(rows)


def split_by_patient(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """60/20/20 train/val/test split. All images of a patient land in the same
    split (avoids leakage) and the split is stratified on subtype
    (normal / bacteria / virus)."""
    df = df.copy()
    df["split"] = "train"
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, df["subtype"], df["patient"]))
    df.loc[df.index[test_idx], "split"] = "test"

    rest = df.iloc[rest_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    _, val_idx = next(inner.split(rest, rest["subtype"], rest["patient"]))
    df.loc[rest.index[val_idx], "split"] = "val"
    return df


# --------------------------------------------------------------------------
# Preprocessing / datasets
# --------------------------------------------------------------------------
def load_rgb(path: str) -> Image.Image:
    # X-rays are single channel; replicate to 3 channels for ImageNet backbones
    return Image.open(path).convert("RGB")


def load_resized(path: str) -> Image.Image:
    """Exactly the deterministic part of the preprocessing done in predict.py."""
    return load_rgb(path).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def get_transforms(train: bool, augment: bool = True) -> transforms.Compose:
    # images are already resized to IMG_SIZE x IMG_SIZE by load_resized()
    base = []
    if train and augment:
        base += [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    base += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(base)


class XrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype("float32").tolist()
        self.transform = transform
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        if i not in self._cache:  # dataset is small: cache decoded + resized images
            self._cache[i] = load_resized(self.paths[i])
        img = self._cache[i]
        return self.transform(img), torch.tensor(self.labels[i])


def make_loader(df, train: bool, augment: bool = True, batch_size: int = 32) -> DataLoader:
    ds = XrayDataset(df, get_transforms(train, augment))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
BACKBONES = {
    # name: (constructor, ImageNet weights)
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}


class XrayNet(nn.Module):
    """Pre-trained backbone + small custom head producing a single logit."""

    def __init__(self, backbone: str, inner_size: int = 0, dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        ctor, weights = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        if backbone == "resnet18":
            n_features = net.fc.in_features
            net.fc = nn.Identity()
            self.features = net
        else:
            # mobilenet / efficientnet: features -> avgpool -> classifier
            linear = next(m for m in net.classifier if isinstance(m, nn.Linear))
            n_features = linear.in_features
            self.features = nn.Sequential(net.features, net.avgpool, nn.Flatten())
        head: list[nn.Module] = []
        if inner_size > 0:
            head += [nn.Linear(n_features, inner_size), nn.ReLU()]
            n_features = inner_size
        if dropout > 0:
            head += [nn.Dropout(dropout)]
        head += [nn.Linear(n_features, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


# --------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    backbone: str = "mobilenet_v3_large"
    freeze_backbone: bool = False
    learning_rate: float = 3e-4
    inner_size: int = 0
    dropout: float = 0.0
    augment: bool = True
    epochs: int = 12
    batch_size: int = 16
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = SEED


@dataclass
class TrainResult:
    config: TrainConfig
    history: list = field(default_factory=list)
    best_epoch: int = 0
    best_val_auc: float = 0.0
    seconds: float = 0.0

    def summary(self) -> dict:
        return {**asdict(self.config), "best_epoch": self.best_epoch,
                "val_auc": round(self.best_val_auc, 4), "seconds": round(self.seconds, 1)}


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_model(cfg: TrainConfig, train_df, val_df, device=None, verbose: bool = True):
    """Train with AdamW + BCE, early stopping on validation ROC-AUC.
    Returns (model with best weights, TrainResult)."""
    device = device or get_device()
    seed_everything(cfg.seed)
    model = XrayNet(cfg.backbone, cfg.inner_size, cfg.dropout).to(device)
    if cfg.freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False

    train_loader = make_loader(train_df, train=True, augment=cfg.augment, batch_size=cfg.batch_size)
    val_loader = make_loader(val_df, train=False, batch_size=cfg.batch_size)

    # class imbalance (~62% pneumonia): weight positives down
    n_pos = train_df["label"].sum()
    pos_weight = torch.tensor(float((len(train_df) - n_pos) / n_pos), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    result = TrainResult(config=cfg)
    best_state, bad_epochs, t0 = None, 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
        scheduler.step()

        y_val, p_val = predict_proba(model, val_loader, device)
        val_loss = nn.functional.binary_cross_entropy(torch.tensor(p_val), torch.tensor(y_val)).item()
        m = metrics(y_val, p_val)
        row = {"epoch": epoch, "train_loss": running / len(train_df), "val_loss": val_loss,
               "val_auc": m["roc_auc"], "val_acc": m["accuracy"]}
        result.history.append(row)
        if verbose:
            print(f"  epoch {epoch:2d} train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
                  f"val_auc={m['roc_auc']:.4f} val_acc={m['accuracy']:.4f}")

        if m["roc_auc"] > result.best_val_auc + 1e-4:
            result.best_val_auc, result.best_epoch = m["roc_auc"], epoch
            best_state, bad_epochs = copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    result.seconds = time.time() - t0
    return model, result


def choose_threshold(y_true, y_prob, beta: float = 2.0, max_threshold: float = 0.5) -> float:
    """Pick the decision threshold that maximises F-beta on validation data.

    beta=2 weights recall twice as much as precision: in a triage setting a
    missed pneumonia is costlier than a false alarm. Only thresholds <= 0.5
    are considered, i.e. tuning may make the model *more* sensitive than the
    default but never less (with ~120 validation images, raising the
    threshold to squeeze out precision overfits and costs recall on new data).
    Ties go to the highest threshold.
    """
    from sklearn.metrics import fbeta_score

    best_t, best_f = 0.5, -1.0
    for t in np.round(np.arange(0.05, max_threshold + 1e-9, 0.01), 2):
        f = fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta, zero_division=0)
        if f >= best_f:
            best_t, best_f = float(t), f
    return best_t
