from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
import torch
from collections import defaultdict
from transforms import (
    train_baseline_transform,
    train_augmented_transform,
    eval_transform
)

BATCH_SIZE = 32
SEED = 42
DATASET_ROOT =Path("../data/dataset")
TRAIN_ROOT = DATASET_ROOT / "cleaned" / "train"
VAL_ROOT = DATASET_ROOT / "cleaned" / "val"
TEST_ROOT = DATASET_ROOT / "test"

generator = torch.Generator().manual_seed(SEED)

train_dataset_baseline = datasets.ImageFolder(root=TRAIN_ROOT,transform=train_baseline_transform)

train_dataset_augmented = datasets.ImageFolder(root=TRAIN_ROOT,transform=train_augmented_transform)

val_dataset = datasets.ImageFolder(root=VAL_ROOT,transform=eval_transform)

test_dataset = datasets.ImageFolder(root=TEST_ROOT,transform=eval_transform)

# ===================
# DataLoaders
# ===================
train_loader_baseline = DataLoader(
    train_dataset_baseline,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
train_loader_augmented = DataLoader(
    train_dataset_augmented,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ===================
# sanity check
# ===================

assert train_dataset_baseline.classes == train_dataset_augmented.classes
assert train_dataset_baseline.classes == val_dataset.classes
assert train_dataset_baseline.classes == test_dataset.classes

assert train_dataset_baseline.class_to_idx == train_dataset_augmented.class_to_idx
assert train_dataset_baseline.class_to_idx == val_dataset.class_to_idx
assert train_dataset_baseline.class_to_idx == test_dataset.class_to_idx

assert len(train_dataset_baseline.classes) == 8

print("Classes:")
print(train_dataset_baseline.classes)

print("\nClass mapping:")
print(train_dataset_baseline.class_to_idx)

print("\nDataset sizes:")
print(f"Train: {len(train_dataset_baseline)}")
print(f"Val:   {len(val_dataset)}")
print(f"Test:  {len(test_dataset)}")

# ===================
# save SEED for reproducibility
# ===================

torch.save({"seed": SEED,},"seed42.pt")





