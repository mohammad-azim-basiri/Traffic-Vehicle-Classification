from symtable import Class

from torch.utils.data import DataLoader,Subset
from torchvision import datasets
from torchvision.models import ResNet18_Weights
from pathlib import Path
import torch
from torch.utils.data import Sampler
from collections import defaultdict
from collections import Counter
from transforms import (
    train_baseline_transform,
    train_augmented_transform,
    eval_transform,
    train_transform_resnet
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

# print("Classes:")
# print(train_dataset_baseline.classes)

# print("\nClass mapping:")
# print(train_dataset_baseline.class_to_idx)

# print("\nDataset sizes:")
# print(f"Train: {len(train_dataset_baseline)}")
# print(f"Val:   {len(val_dataset)}")
# print(f"Test:  {len(test_dataset)}")

# ===================
# save SEED for reproducibility
# ===================

torch.save({"seed": SEED,},"seed42.pt")

# print(Counter(train_dataset_baseline.targets)) #Counter({7: 119, 2: 80, 5: 80, 6: 80, 1: 79, 4: 78, 3: 77, 0: 74})


# ===================================
# Imbalanced dataset
#====================================
imbalance_counts = {
    0: 34,   # ambulance
    1: 39,   # autobus
    2: 40,   # kamyun
    3: 77,   # kamyunet
    4: 38,   # minibus
    5: 80,   # savari
    6: 80,   # taxi
    7: 119,  # vanet
}
def create_imbalanced_indices(dataset, target_counts, seed=SEED):
    generator = torch.Generator().manual_seed(seed)

    class_indices = defaultdict(list)

    for idx, class_id in enumerate(dataset.targets):
        class_indices[class_id].append(idx)

    retained_indices = []

    for class_id, indices in class_indices.items():
        indices = torch.tensor(indices)

        shuffled = indices[
            torch.randperm(len(indices), generator=generator)
        ]

        n = target_counts[class_id]

        if n > len(indices):
            raise ValueError(
                f"Class {class_id} has only {len(indices)} samples, "
                f"but target is {n}."
            )

        retained_indices.extend(shuffled[:n].tolist())

    return retained_indices

# -------------------------
# Standard sampling
# -------------------------
imbalanced_indices = create_imbalanced_indices(train_dataset_baseline, imbalance_counts,SEED)
# print(len(imbalanced_indices))

imbalanced_train_dataset = Subset(train_dataset_baseline, imbalanced_indices)
imbalanced_targets =[
    train_dataset_baseline.targets[i] for i in imbalanced_indices
]
train_loader_standard_imbalanced = DataLoader(
    imbalanced_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
#print(len(imbalanced_targets))
#print(Counter(imbalanced_targets))

# -------------------------
# Balanced sampling
# -------------------------
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size,seed=SEED):
        if batch_size % 8 != 0 :
            raise ValueError("batch_size must be divisible by 8.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.num_classes = 8
        self.samples_per_class = batch_size // self.num_classes
        class_indices = defaultdict(list)

        for subset_idx, original_idx in enumerate(dataset.indices):
            class_id = dataset.dataset.targets[original_idx]
            class_indices[class_id].append(subset_idx)

        self.class_indices = {
            class_id:torch.tensor(indices) for class_id, indices in class_indices.items()
        }
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)

        for _ in range(self.num_batches):
            batch = []
            for class_id in range(self.num_classes):
                indices = self.class_indices[class_id]
                selected = torch.randint(
                    low=0,
                    high=len(indices),
                    size=(self.samples_per_class,),
                    generator=generator
                )
                batch.extend(indices[selected].tolist())
            yield batch

    def __len__(self):
        return self.num_batches


balanced_sampler = BalancedBatchSampler(
    imbalanced_train_dataset,
    batch_size=BATCH_SIZE,
    seed=SEED
)
train_loader_balanced_imbalanced  = DataLoader(
    imbalanced_train_dataset,
    batch_sampler=balanced_sampler
)

images, labels = next(iter(train_loader_balanced_imbalanced ))
#print(f"labels:{labels}")  #labels:tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,6, 6, 6, 6, 7, 7, 7, 7])
#print(Counter(labels.tolist()))  #Counter({0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4})


for batch_idx, (_, labels) in enumerate(train_loader_balanced_imbalanced ):
    counts = Counter(labels.tolist())
    #print(f"Batch {batch_idx}: {counts}")
    if batch_idx == 4:
        break



weights = ResNet18_Weights.DEFAULT
transform_resnet = weights.transforms()

train_dataset_resnet = datasets.ImageFolder(root=TRAIN_ROOT,transform=transform_resnet)
val_dataset_resnet = datasets.ImageFolder(root=VAL_ROOT,transform=transform_resnet)
test_dataset_resnet = datasets.ImageFolder(root=TEST_ROOT,transform=transform_resnet)

train_loader_resnet = DataLoader(
    train_dataset_resnet,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader_resnet = DataLoader(
    val_dataset_resnet,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
test_loader_resnet = DataLoader(
    test_dataset_resnet,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)
assert train_dataset_resnet.class_to_idx == val_dataset_resnet.class_to_idx
assert train_dataset_resnet.class_to_idx == test_dataset_resnet.class_to_idx