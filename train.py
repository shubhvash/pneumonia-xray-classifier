"""Train the final pneumonia classifier and export it to ONNX.

The hyper-parameters below are the best configuration found in notebook.ipynb.

Usage:
    uv run python train.py                     # train with the selected config
    uv run python train.py --epochs 5 --lr 1e-3  # override anything

Outputs (in models/):
    pneumonia_classifier.onnx   model used by the web service
    metadata.json               preprocessing constants, threshold, test metrics
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import common
from common import TrainConfig

MODELS_DIR = Path(__file__).parent / "models"
ONNX_PATH = MODELS_DIR / "pneumonia_classifier.onnx"
META_PATH = MODELS_DIR / "metadata.json"


def parse_args() -> argparse.Namespace:
    d = TrainConfig(
        # ---- selected in notebook.ipynb (see reports/experiments.csv) ----
        backbone="mobilenet_v3_large",
        freeze_backbone=False,
        learning_rate=3e-4,
        inner_size=64,
        dropout=0.5,
        augment=True,
        epochs=10,
        patience=3,
    )
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default=d.backbone, choices=list(common.BACKBONES))
    p.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=d.freeze_backbone)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--inner-size", type=int, default=d.inner_size)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=d.augment)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    return p.parse_args()


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, common.IMG_SIZE, common.IMG_SIZE)
    torch.onnx.export(
        model, (dummy,), str(path),
        input_names=["input"], output_names=["logit"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}},
        dynamo=True, external_data=False,
    )


def check_onnx_matches_torch(model: torch.nn.Module, path: Path) -> float:
    model = model.cpu().eval()
    x = torch.randn(4, 3, common.IMG_SIZE, common.IMG_SIZE)
    with torch.no_grad():
        expected = model(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    return float(np.abs(expected - got).max())


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        backbone=args.backbone, freeze_backbone=args.freeze_backbone, learning_rate=args.lr,
        inner_size=args.inner_size, dropout=args.dropout, augment=args.augment,
        epochs=args.epochs, patience=args.patience, seed=args.seed, batch_size=args.batch_size,
    )
    device = common.get_device()
    print(f"device: {device}\nconfig: {cfg}")

    df = common.split_by_patient(common.build_catalogue(), seed=cfg.seed)
    train_df, val_df, test_df = (df[df.split == s] for s in ["train", "val", "test"])
    print(f"images: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    model, result = common.train_model(cfg, train_df, val_df, device=device)
    print(f"best epoch {result.best_epoch}, val ROC-AUC {result.best_val_auc:.4f}")

    y_val, p_val = common.predict_proba(model, common.make_loader(val_df, train=False), device)
    threshold = common.choose_threshold(y_val, p_val)
    y_test, p_test = common.predict_proba(model, common.make_loader(test_df, train=False), device)
    test_metrics = {k: round(float(v), 4) for k, v in common.metrics(y_test, p_test, threshold).items()}
    print(f"decision threshold: {threshold}")
    print(f"test metrics: {test_metrics}")

    MODELS_DIR.mkdir(exist_ok=True)
    export_onnx(model, ONNX_PATH)
    diff = check_onnx_matches_torch(model, ONNX_PATH)
    print(f"ONNX export -> {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB), max |torch - onnx| = {diff:.2e}")
    assert diff < 1e-3, "ONNX output does not match PyTorch output"

    metadata = {
        "model_file": ONNX_PATH.name,
        "class_names": common.CLASS_NAMES,
        "img_size": common.IMG_SIZE,
        "mean": common.IMAGENET_MEAN,
        "std": common.IMAGENET_STD,
        "threshold": threshold,
        "train_config": asdict(cfg),
        "best_epoch": result.best_epoch,
        "val_roc_auc": round(result.best_val_auc, 4),
        "test_metrics": test_metrics,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
