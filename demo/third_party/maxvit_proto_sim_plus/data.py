#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Loading Module
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T

CLS_14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
CLS14_TO_IDX = {c: i for i, c in enumerate(CLS_14)}

DEFAULT_CLASS_WEIGHTS = {
    "Atelectasis": 1.0,
    "Cardiomegaly": 3.0,
    "Effusion": 1.0,
    "Infiltration": 1.0,
    "Mass": 2.0,
    "Nodule": 2.0,
    "Pneumonia": 5.0,
    "Pneumothorax": 2.0,
    "Consolidation": 2.0,
    "Edema": 3.0,
    "Emphysema": 3.0,
    "Fibrosis": 4.0,
    "Pleural_Thickening": 3.0,
    "Hernia": 8.0,
}


def extract_patient_id(image_name):
    name = image_name.replace('.png', '').replace('.jpg', '')
    parts = name.split('_')
    if len(parts) >= 1:
        return parts[0]
    return name


def labels_to_multihot(label_str, class_to_idx, n_cls=14):
    y = np.zeros(n_cls, dtype=np.float32)
    if pd.isna(label_str):
        return y
    for t in str(label_str).split("|"):
        t = t.strip()
        if t in class_to_idx:
            y[class_to_idx[t]] = 1.0
    return y


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class ChestXrayDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        img_root: Path,
        transform=None,
        return_path: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / row["Image Index"]

        img = load_rgb(img_path)
        if self.transform:
            img = self.transform(img)

        label = row["y14"]

        if self.return_path:
            return img, torch.from_numpy(label), str(img_path)
        return img, torch.from_numpy(label)


def prepare_data_splits(config: dict):
    project_root = Path(config["paths"]["project_root"])
    img_root = Path(config["paths"]["image_root"])
    csv_entry = Path(config["paths"]["csv_entry"])
    test_list_path = Path(config["paths"]["test_list"])
    train_val_list_path = Path(config["paths"]["train_val_list"])

    entry_df = pd.read_csv(csv_entry)
    entry_df = entry_df[["Image Index", "Finding Labels"]].copy()
    entry_df["y14"] = entry_df["Finding Labels"].apply(
        lambda s: labels_to_multihot(s, CLS14_TO_IDX, 14)
    )

    with open(test_list_path, "r") as f:
        test_files = set(line.strip() for line in f if line.strip())

    with open(train_val_list_path, "r") as f:
        train_val_files = set(line.strip() for line in f if line.strip())

    all_imgs = set(p.name for p in img_root.glob("*.png"))
    entry_df = entry_df[entry_df["Image Index"].isin(all_imgs)].reset_index(drop=True)

    test_df = entry_df[entry_df["Image Index"].isin(test_files)].reset_index(drop=True)

    train_val_df = entry_df[
        (entry_df["Image Index"].isin(train_val_files)) &
        (~entry_df["Image Index"].isin(test_files))
    ].reset_index(drop=True)

    print(f"Total images in CSV: {len(entry_df)}")
    print(f"Test images found: {len(test_df)}")
    print(f"Train+Val images found: {len(train_val_df)}")

    train_val_df["patient_id"] = train_val_df["Image Index"].apply(extract_patient_id)

    np.random.seed(42)
    unique_patients = list(train_val_df["patient_id"].unique())
    np.random.shuffle(unique_patients)
    val_patient_count = int(len(unique_patients) * 0.2)
    val_patients = set(unique_patients[:val_patient_count])
    train_patients = set(unique_patients[val_patient_count:])

    train_df = train_val_df[train_val_df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = train_val_df[train_val_df["patient_id"].isin(val_patients)].reset_index(drop=True)

    print(f"\nData splits (by patient ID):")
    print(f"  Train: {len(train_df)} images, {len(train_patients)} patients")
    print(f"  Val: {len(val_df)} images, {len(val_patients)} patients")
    print(f"  Test: {len(test_df)} images")

    print(f"\nClass distribution in training set:")
    for i, cls in enumerate(CLS_14):
        count = int(train_df["y14"].apply(lambda x: x[i]).sum())
        print(f"  {cls:20s}: {count:5d}")

    return train_df, val_df, test_df


def get_transforms(img_size: int, is_train: bool = True):
    if is_train:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def compute_class_weights(df: pd.DataFrame, n_cls: int = 14) -> torch.Tensor:
    pos_counts = np.zeros(n_cls)
    for i in range(n_cls):
        pos_counts[i] = df["y14"].apply(lambda x: x[i]).sum()

    total = len(df)
    weights = total / (n_cls * pos_counts + 1e-6)
    weights = weights / weights.mean()

    weights = np.clip(weights, 1.0, 4.0)

    return torch.FloatTensor(weights)


if __name__ == "__main__":
    with open(r"D:\统计建模\mavit_proto_sim_plus\config.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_df, val_df, test_df = prepare_data_splits(config)

    print("\nClass distribution in training set:")
    for i, cls in enumerate(CLS_14):
        count = int(train_df["y14"].apply(lambda x: x[i]).sum())
        print(f"  {cls:20s}: {count:5d}")

    weights = compute_class_weights(train_df)
    print(f"\nClass weights: {weights}")
