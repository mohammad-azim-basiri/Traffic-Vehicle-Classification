from torch.utils.data import Subset, DataLoader,Dataset
from torchvision import datasets
from pathlib import Path
import torch
from collections import Counter

from transforms import (
    train_baseline_transform,
    train_augmented_transform,
    eval_transform
)


SEED = 42
DATASET_ROOT =Path("../data/dataset")


dataset_base = datasets.ImageFolder(
    root = DATASET_ROOT / "train",
    transform = None
)

generator = torch.Generator().manual_seed(SEED)

train_indices = []
val_indices = []

targets = torch.tensor(dataset_base.targets)

for class_id in range(len(dataset_base.classes)):
    class_indices = torch.where(targets == class_id)[0]

    perm = torch.randperm(
        len(class_indices),
        generator=generator
    )

    class_indices = class_indices[perm].tolist()

    train_indices.extend(class_indices[:40])
    val_indices.extend(class_indices[40:50])



class TransformSubset(Dataset):

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        image, label = self.dataset[self.indices[idx]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label



train_dataset_baseline = TransformSubset(dataset_base, train_indices, train_baseline_transform)

train_dataset_augmented = TransformSubset(dataset_base, train_indices, train_augmented_transform)

val_dataset = TransformSubset(dataset_base, val_indices, eval_transform)



# ===================
# DataLoaders
# ===================

train_loader_baseline = DataLoader(
    train_dataset_baseline,
    batch_size=32,
    shuffle=True
)
train_loader_augmented = DataLoader(
    train_dataset_augmented,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# ===================
# sanity check
# ===================

train_labels = [dataset_base.targets[i] for i in train_indices]
val_labels = [dataset_base.targets[i] for i in val_indices]

train_counts = Counter(train_labels)
val_counts = Counter(val_labels)

for class_id, class_name in enumerate(dataset_base.classes):

    assert train_counts[class_id] == 40, (
        f"Class '{class_name}': expected 40 train samples, "
        f"got {train_counts[class_id]}"
    )

    assert val_counts[class_id] == 10, (
        f"Class '{class_name}': expected 10 validation samples, "
        f"got {val_counts[class_id]}"
    )

# ===================
# save indices and SEED for reproducibility
# ===================

torch.save(
    {
        "seed": SEED,
        "train_indices": train_indices,
        "val_indices": val_indices,
    },
    "split_seed42.pt"
)

# ===================
# load info we saved before
# ===================

# split = torch.load("split_seed42.pt")
# seed = split["seed"]
# train_indices = split["train_indices"]
# val_indices = split["val_indices"]
