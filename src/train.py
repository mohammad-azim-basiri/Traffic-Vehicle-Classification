import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import resnet18,ResNet18_Weights,vgg19,mobilenet_v3_small,MobileNet_V3_Small_Weights

from models import (
    SmallCnn,
    resnet18_model,
    mobilenet_v3_small_model,
    build_mobilenet_v3_large,
    DepthwiseCNN
)
from utils import run_experiment,run_experiment_bce,set_seed,plot_training_history,create_scheduler
import time
from pathlib import Path

from datasets import (
train_loader_baseline,
train_loader_augmented,
val_loader,
train_loader_standard_imbalanced,
train_loader_balanced_imbalanced,
train_loader_resnet,
val_loader_resnet
)

SEED = 42
set_seed(SEED)

output_dir = Path("../results/img")
output_dir.mkdir(parents=True, exist_ok=True)

num_epochs = 50
LR=0.001
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
    set_seed(SEED)
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

# ==========================================
# weight decay experiments
# ==========================================
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

    start_time = time.time()

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
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

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

# ===========================================
# train_small_cnn_bce experiments
# ===========================================
def train_small_cnn_bce(dropout=0.0,pooling="max"):
    print(f"=== Training Small CNN | BCE ===")

    set_seed(SEED)
    model_bce = SmallCnn(num_classes=8,dropout=dropout,pooling=pooling).to(device)
    optimizer_bce = optim.Adam(model_bce.parameters(), lr=0.001)

    start_time = time.time()
    history_bce = run_experiment_bce(
        model=model_bce,
        train_loader=train_loader_baseline,
        val_loader=val_loader,
        optimizer=optimizer_bce,
        device=device,
        epochs=num_epochs,
        checkpoint_path=f"best_bce_small_cnn.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history_bce,
        f"best_bce_small_cnn",
        output_dir
    )
    return model_bce,history_bce

# ===========================================
# train RESNET18 model
# ===========================================
def train_resnet18(num_classes=8):
    set_seed(SEED)
    model_resnet = resnet18_model(num_classes=num_classes)
    optimizer = torch.optim.Adam(model_resnet.fc.parameters(),lr=LR)

    start_time = time.time()
    history = run_experiment(
        model_resnet,
        train_loader_resnet,
        val_loader_resnet,
        optimizer,
        device,
        num_epochs,
        "resnet18_pretrained.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history,
        "resnet18_pretrained",
        output_dir
    )
    return model_resnet

# ===========================================
# train RESNET18 model fine-tuned
# ===========================================
def train_resnet18_ft(num_classes=8):
    set_seed(SEED)
    model_resnet_ft = resnet18_model(num_classes=num_classes)

    model_resnet_ft.load_state_dict(
        torch.load(
            "../results/saved/resnet18_pretrained.pt",
            map_location=device
        )
    )
    for param in model_resnet_ft.parameters():
        param.requires_grad = False

    for param in model_resnet_ft.layer4.parameters():
        param.requires_grad = True

    for param in model_resnet_ft.fc.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model_resnet_ft.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_resnet_ft.parameters())

    print(f"Trainable params: {trainable_params}")
    print(f"Total params: {total_params}")

    optimizer_resnet_ft = optim.Adam([
        {'params': model_resnet_ft.layer4.parameters(), 'lr': 1e-4},
        {'params': model_resnet_ft.fc.parameters(), 'lr': 1e-3},
    ])
    start_time = time.time()
    history_resnet_ft = run_experiment(
        model=model_resnet_ft,
        train_loader=train_loader_resnet,
        val_loader=val_loader_resnet,
        optimizer=optimizer_resnet_ft,
        device=device,
        epochs=num_epochs,
        checkpoint_path=f"best_resnet18_ft.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    plot_training_history(
        history_resnet_ft,
        "resnet18_fine-tuned",
        output_dir

    )
    return history_resnet_ft

# ====================================
# train mobilenet_v3_small_model
# ====================================
def train_mobilenet_v3_small(num_classes=8,transform="base",train_load=train_loader_resnet,val_load=val_loader_resnet):
    print("=== Training MobileNet v3 small model ===")
    set_seed(SEED)
    model_mobilenet_small = mobilenet_v3_small_model(num_classes=8).to(device)

    print(device)
    total_params = sum(p.numel() for p in model_mobilenet_small.parameters())
    trainable_params = sum(p.numel() for p in model_mobilenet_small.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")  # 1_526_056
    print(f"Trainable parameters: {trainable_params:,}")  # 1_526_056

    for name, module in model_mobilenet_small.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:15} {params:,}")

    start_time = time.time()
    optimizer = torch.optim.AdamW(model_mobilenet_small.parameters(), lr=1e-3, weight_decay=1e-4)
    history = run_experiment(
        model_mobilenet_small,
        train_load,
        val_load,
        optimizer,
        device,
        epochs= num_epochs,
        checkpoint_path=f"mobilenet_v3_small_model_{transform}.pt",
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    plot_training_history(
        history,
        f"mobilenet_v3_small_model_{transform}",
        output_dir
    )
    return history

# ====================================
# train mobilenet_v3_large_model
# ====================================
def train_mobilenet_v3_large(
    num_classes=8,
    transform="base",
    train_load=train_loader_resnet,
    val_load=val_loader_resnet,
    mode="full",
    lr=1e-3,
    unfreeze_last_blocks=2
):
    print("=== Training MobileNet v3 large model ===")
    set_seed(SEED)
    model_mobilenet_large = build_mobilenet_v3_large(num_classes=num_classes).to(device)

    if mode == "full":
        for param in model_mobilenet_large.parameters():
            param.requires_grad = True
    elif mode == "head":
        for param in model_mobilenet_large.parameters():
            param.requires_grad = False
        for param in model_mobilenet_large.classifier.parameters():
            param.requires_grad = True
    elif mode == "last_blocks":
        for param in model_mobilenet_large.parameters():
            param.requires_grad = False
        for param in model_mobilenet_large.features[-unfreeze_last_blocks:].parameters():
            param.requires_grad = True
        for param in model_mobilenet_large.classifier.parameters():
            param.requires_grad = True

    else:
        raise ValueError("mode must be 'full', 'head' or 'last_blocks'")

    total_params = sum(p.numel() for p in model_mobilenet_large.parameters())
    trainable_params = sum(p.numel() for p in model_mobilenet_large.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model_mobilenet_large.parameters()
        ),
        lr=lr,
        weight_decay=1e-4
    )

    start_time = time.time()
    history = run_experiment(
        model_mobilenet_large,
        train_load,
        val_load,
        optimizer,
        device,
        epochs=num_epochs,
        checkpoint_path=f"mobilenet_v3_large_{mode}_{transform}.pt",
    )
    end_time = time.time()
    print(f"Training time:{end_time - start_time:.2f} seconds")
    plot_training_history(
        history,
        f"mobilenet_v3_large_model_{mode}_{transform}",
        output_dir
    )
    return history



def train_depthwise_model(num_classes=8,dropout=0.0):
    print(f"=== train depthwise model ===")

    set_seed(SEED)
    model = DepthwiseCNN(num_classes=num_classes,dropout=dropout).to(device)
    optimizer_depth = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    history = run_experiment(
        model=model,
        train_loader=train_loader_baseline,
        val_loader=val_loader,
        optimizer=optimizer_depth,
        device=device,
        epochs=num_epochs,
        checkpoint_path=f"best_depthwise_{dropout}.pt",
        scheduler=None
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    plot_training_history(
        history,
        f"depthwise_model_{dropout}",
        output_dir
    )
    return model,history


if __name__ == "__main__":
    train_small_cnn()
    train_augmented()
    dropout_exp()
    train_small_cnn_avgpool(pooling="avg")
    weight_decay_exp()
    train_scheduler()
    reducelronplateau_exp()

    train_standard_imbalanced()
    train_balanced_imbalanced()

    train_small_cnn_bce()

    train_resnet18(8)
    train_resnet18_ft(8)

    train_mobilenet_v3_small(num_classes=8,transform="base",train_load=train_loader_resnet,val_load=val_loader_resnet)
    train_mobilenet_v3_small(num_classes=8,transform="better_augment",train_load=train_loader_baseline,val_load=val_loader)

    train_mobilenet_v3_large(num_classes=8,transform="base",train_load=train_loader_resnet,val_load=val_loader_resnet,mode="full",lr=1e-3,unfreeze_last_blocks=3)
    train_mobilenet_v3_large(num_classes=8,transform="base",train_load=train_loader_resnet,val_load=val_loader_resnet,mode="head",lr=1e-3,unfreeze_last_blocks=3)
    train_mobilenet_v3_large(num_classes=8,transform="base",train_load=train_loader_resnet,val_load=val_loader_resnet,mode="last_blocks",lr=1e-3,unfreeze_last_blocks=3)

    train_mobilenet_v3_large(num_classes=8, transform="augment", train_load=train_loader_baseline,val_load=val_loader, mode="full", lr=1e-3, unfreeze_last_blocks=3)
    train_mobilenet_v3_large(num_classes=8, transform="augment", train_load=train_loader_baseline,val_load=val_loader, mode="head", lr=1e-3, unfreeze_last_blocks=3)
    train_mobilenet_v3_large(num_classes=8, transform="augment", train_load=train_loader_baseline,val_load=val_loader, mode="last_blocks", lr=1e-3, unfreeze_last_blocks=3)

    train_depthwise_model(num_classes=8,dropout=0.5)