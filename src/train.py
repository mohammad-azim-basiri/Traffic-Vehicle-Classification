import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from models import SmallCnn
from utils import run_experiment,set_seed,plot_training_history,create_scheduler
import time
from pathlib import Path

from datasets import (
train_loader_baseline,
train_loader_augmented,
val_loader,
train_loader_standard_imbalanced,
train_loader_balanced_imbalanced
)

SEED = 42
set_seed(SEED)

output_dir = Path("../results/img")
output_dir.mkdir(parents=True, exist_ok=True)

num_epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# train base model with dropout = 0.0 and MaxPooling function definition
# ============================================
def train_small_cnn(dropout=0.0,pooling="max"):
    print(f"=== Training Small CNN | Dropout={dropout} ===")

    set_seed(SEED)
    model = SmallCnn(num_classes=8,dropout=dropout,pooling=pooling).to(device)
    optimizer_baseline = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    # checkpoint_dir = Path("../results/saved")
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = run_experiment(
        model=model,
        train_loader=train_loader_baseline,
        val_loader=val_loader,
        optimizer=optimizer_baseline,
        device=device,
        epochs=num_epochs,
        checkpoint_path=f"best_baseline_small_cnn_{dropout}.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history,
        f"baseline_small_cnn_{dropout}",
        output_dir
    )
    return model,history

# =================================================
# train augmented experiment function definition
# =================================================
def train_augmented():
    print("=== Training Small CNN - Augmented ===")

    start = time.time()
    set_seed(SEED)
    model = SmallCnn(num_classes=8,dropout=0).to(device)
    optimizer_augmented = optim.Adam(model.parameters(), lr=0.001)

    # checkpoint_dir = Path("../results/saved")
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = run_experiment(
        model=model,
        train_loader=train_loader_augmented,
        val_loader=val_loader,
        optimizer=optimizer_augmented,
        device=device,
        epochs=num_epochs,
        checkpoint_path="best_augmented_small_cnn.pt",
        scheduler=None
    )
    end = time.time()
    print(f"Training time: {end - start:.2f} seconds")

    plot_training_history(
        history,
        "augmented_small_cnn",
        output_dir
    )

    return model,history

# =================================================
# dropout experiment function definition
# =================================================
def dropout_exp():
    print("=== Dropping output experiment ===")
    dropout_values = [0.3, 0.5]
    dropout_results = {}

    for dropout in dropout_values:
        _ , history = train_small_cnn(dropout=dropout)
        dropout_results[dropout] = history


    return dropout_results

# =================================================
# train base model with dropout = 0.0 and AvgPooling function definition
# =================================================
def train_small_cnn_avgpool(dropout=0.0,pooling="avg"):

    set_seed(SEED)
    model_avg = SmallCnn(num_classes=8,dropout=dropout,pooling=pooling).to(device)
    optimizer_avg = optim.Adam(model_avg.parameters(), lr=0.001)

    start_time = time.time()

    # checkpoint_dir = Path("../results/saved")
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_avg = run_experiment(
        model=model_avg,
        train_loader=train_loader_baseline,
        val_loader=val_loader,
        optimizer=optimizer_avg,
        device=device,
        epochs=num_epochs,
        checkpoint_path=f"best_small_cnn_avg_pool.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history_avg,
        f"small_cnn_avg_pool",
        output_dir
    )
    return model_avg,history_avg

# =================================================
# weight decay experiment function definition
# =================================================
def train_small_cnn_weight_decay(dropout=0.0,pooling="max",weight_decay=0.0001):
    print(f"=== Training Small CNN | weight_decay={weight_decay} ===")

    set_seed(SEED)
    model_weight_decay = SmallCnn(num_classes=8,dropout=dropout,pooling=pooling).to(device)
    optimizer_weight_decay = optim.AdamW(model_weight_decay.parameters(), lr=0.001,weight_decay=weight_decay)

    start_time = time.time()

    history_weight_decay = run_experiment(
        model=model_weight_decay,
        train_loader=train_loader_baseline,
        val_loader=val_loader,
        optimizer=optimizer_weight_decay,
        device=device,
        epochs=num_epochs,
        checkpoint_path=f"best_weight_decay_small_cnn_{weight_decay}.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history_weight_decay,
        f"weight_decay_small_cnn_{weight_decay}",
        output_dir
    )
    return model_weight_decay,history_weight_decay

def weight_decay_exp():
    set_seed(SEED)
    weight_decay_values = [0, 1e-4]

    results = {}

    for weight_decay in weight_decay_values:

        model, history = train_small_cnn_weight_decay(
            dropout=0.0,
            pooling="max",
            weight_decay=weight_decay
        )

        results[weight_decay] = history

    return results

# ===========================================
# train small cnn with StepLR
# ===========================================
def train_scheduler():
    print("=== Training Small CNN | StepLR ===")
    set_seed(SEED)

    model = SmallCnn(num_classes=8,dropout=0.0,pooling="max").to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    start_time = time.time()

    history = run_experiment(
        model=model,
        train_loader=train_loader_baseline,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=num_epochs,
        checkpoint_path="best_small_cnn_step_lr.pt",
        scheduler=scheduler
    )
    end_time = time.time()

    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history,
        "small_cnn_step_lr",
        output_dir
    )
    return model, history

# ===========================================
# reducelronplateau_exp
# ===========================================
def reducelronplateau_exp():
    print("=== Training Small CNN | ReduceLROnPlateau ===")
    set_seed(SEED)
    model = SmallCnn(num_classes=8,dropout=0.0,pooling="max").to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = create_scheduler(optimizer)
    history = run_experiment(
        model=model,
        train_loader=train_loader_baseline,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=num_epochs,
        checkpoint_path="best_reduce_lr_small_cnn.pt"
    )

    plot_training_history(
        history,
        "small_cnn_reduce_lr",
        output_dir
    )
    return history, scheduler

# ===========================================
# train_standard_imbalanced
# ===========================================
def train_standard_imbalanced(dropout=0.0,pooling="max"):
    print(f"=== Training Small CNN | train_loader_standard_imbalanced ===")

    set_seed(SEED)
    model_standard_imbalanced = SmallCnn(num_classes=8,dropout=dropout,pooling=pooling).to(device)
    optimizer_standard_imbalanced = optim.Adam(model_standard_imbalanced.parameters(), lr=0.001)

    start_time = time.time()

    history_standard_imbalanced = run_experiment(
        model=model_standard_imbalanced,
        train_loader=train_loader_standard_imbalanced,
        val_loader=val_loader,
        optimizer=optimizer_standard_imbalanced,
        device=device,
        epochs=num_epochs,
        checkpoint_path="train_loader_standard_imbalanced.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history_standard_imbalanced,
        "train_loader_standard_imbalanced",
        output_dir
    )
    return model_standard_imbalanced,history_standard_imbalanced

# ===========================================
# train_loader_balanced_imbalanced
# ===========================================
def train_balanced_imbalanced(dropout=0.0,pooling="max"):
    print(f"=== Training Small CNN | train_balanced_imbalanced ===")

    set_seed(SEED)
    model_balanced_imbalanced = SmallCnn(num_classes=8,dropout=dropout,pooling=pooling).to(device)
    optimizer_balanced_imbalanced = optim.Adam(model_balanced_imbalanced.parameters(), lr=0.001)

    start_time = time.time()

    history_balanced_imbalanced = run_experiment(
        model=model_balanced_imbalanced,
        train_loader=train_loader_balanced_imbalanced,
        val_loader=val_loader,
        optimizer=optimizer_balanced_imbalanced,
        device=device,
        epochs=num_epochs,
        checkpoint_path="train_balanced_imbalanced.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history_balanced_imbalanced,
        "train_balanced_imbalanced",
        output_dir
    )
    return model_balanced_imbalanced,history_balanced_imbalanced

if __name__ == "__main__":
    train_small_cnn()
    # train_augmented()
    # dropout_exp()
    # train_small_cnn_avgpool(pooling="avg")
    # weight_decay_exp()
    # train_scheduler()
    # reducelronplateau_exp()
    # train_standard_imbalanced()
    # train_balanced_imbalanced()