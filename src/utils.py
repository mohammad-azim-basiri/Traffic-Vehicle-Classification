import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import time
import gc
import platform
import subprocess
from datasets import UnlabeledImageDataset
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

criterion = nn.CrossEntropyLoss()
bce_criterion = nn.BCEWithLogitsLoss() # also do sigmoid.
# bce_criterion = nn.BCELoss() # we should do sigmoid.

def run_one_epoch(model, loader, optimizer=None, device="cpu"):
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": running_loss / total,
        "Accuracy": correct / total,
    }


def run_experiment(model,train_loader,val_loader,optimizer,device,epochs=5,checkpoint_path="best_model.pt",scheduler=None):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_val_acc = -float("inf")

    for epoch in range(epochs):
        train_metrics = run_one_epoch( model, loader=train_loader,optimizer=optimizer,device=device)
        val_metrics = run_one_epoch(model,loader=val_loader,optimizer=None,device=device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["Accuracy"])
        history["val_accuracy"].append(val_metrics["Accuracy"])

        if val_metrics["Accuracy"] > best_val_acc:
            best_val_acc = val_metrics["Accuracy"]

            torch.save(
                model.state_dict(),
                f"../results/saved/{checkpoint_path}"
            )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss: {train_metrics['loss']:.3f} | "
            f"val loss: {val_metrics['loss']:.3f} | "
            f"train acc: {train_metrics['Accuracy']:.1%} | "
            f"val acc: {val_metrics['Accuracy']:.1%} | "
            f"current lr: {current_lr}"
        )

        if scheduler is not None:
            if isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_metrics["loss"])

            else:
                scheduler.step()

        if (epoch + 1) % 10 == 0 and (epoch + 1) < epochs:
            for remaining in range(150, 0, -1):
                print(f"\rCooling down... {remaining:03d} seconds remaining", end="")
                time.sleep(1)
            print("\n")

    return history


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # we have error from below line🧐🔍
    # torch.use_deterministic_algorithms(True)


def evaluate_model(model, loader, device="cpu"):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:
            images ,labels = images.to(device) ,labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1).cpu().tolist()
            labels = labels.cpu().tolist()

            all_labels.extend(labels)
            all_predictions.extend(predictions)

    return all_labels, all_predictions


def calculate_classification_metrics(y_true, y_pred):
    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro"
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    per_class_precision = precision_score(
        y_true,
        y_pred,
        average=None
    )

    per_class_recall = recall_score(
        y_true,
        y_pred,
        average=None
    )

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None
    )

    return {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }


def plot_training_history(history, experiment_name, output_dir):
    plt.figure()

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{experiment_name} - Loss")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{experiment_name}_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    plt.figure()

    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name} - Accuracy")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{experiment_name}_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


def plot_confusion_matrices(y_true,y_pred,classes,experiment_name,output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    disp.plot(xticks_rotation=45)

    plt.title(f"{experiment_name} - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{experiment_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=classes
    )

    disp.plot(xticks_rotation=45)

    plt.title(
        f"{experiment_name} - Row-Normalized Confusion Matrix"
    )
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{experiment_name}_confusion_matrix_normalized.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return cm, cm_normalized


def create_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.1,
        min_lr=1e-6
    )


def print_metrics(metrics, classes):
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro: {metrics['recall_macro']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")

    print("\nPer-class metrics:")

    for cls, precision, recall, f1 in zip(
        classes,
        metrics['per_class_precision'],
        metrics['per_class_recall'],
        metrics['per_class_f1']
    ):
        print(
            f"{cls:10s} | "
            f"Precision: {precision:.2f} | "
            f"Recall: {recall:.2f} | "
            f"F1: {f1:.2f}"
        )


def run_one_epoch_bce(model, loader, optimizer=None, device="cpu"):
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            targets = F.one_hot(
                labels,
                num_classes=outputs.size(1)
            ).float()

            loss = bce_criterion(outputs, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": running_loss / total,
        "Accuracy": correct / total,
    }


def run_experiment_bce(model,train_loader,val_loader,optimizer,device,epochs=5,checkpoint_path="best_model.pt",scheduler=None):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_val_acc = -float("inf")

    for epoch in range(epochs):
        train_metrics = run_one_epoch_bce( model, loader=train_loader,optimizer=optimizer,device=device)
        val_metrics = run_one_epoch_bce(model,loader=val_loader,optimizer=None,device=device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["Accuracy"])
        history["val_accuracy"].append(val_metrics["Accuracy"])

        if val_metrics["Accuracy"] > best_val_acc:
            best_val_acc = val_metrics["Accuracy"]

            torch.save(
                model.state_dict(),
                f"../results/saved/{checkpoint_path}"
            )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss: {train_metrics['loss']:.3f} | "
            f"val loss: {val_metrics['loss']:.3f} | "
            f"train acc: {train_metrics['Accuracy']:.1%} | "
            f"val acc: {val_metrics['Accuracy']:.1%} | "
            f"current lr: {current_lr}"
        )

        if scheduler is not None:
            if isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_metrics["loss"])

            else:
                scheduler.step()

    return history


def extract_predictions_and_confidence(model,loader,device="cpu",loss_type="ce"):
    model.eval()
    all_targets = []
    all_predictions = []
    all_confidences = []
    all_scores = []


    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            if loss_type == "ce":
                scores = torch.softmax(logits, dim=1)
            elif loss_type == "bce":
                scores = torch.sigmoid(logits)
            else :
                raise ValueError("loss_type must be either 'ce' or 'bce'")

            predictions = logits.argmax(dim=1)

            confidences = scores.gather(
                dim=1,
                index=predictions.unsqueeze(dim=1)
            ).squeeze(1)
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    return all_targets, all_predictions, all_confidences,all_scores

def cuda_cooldown(seconds=300):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Cooling down for {seconds} seconds...")
    time.sleep(seconds)



# ======================================
# Below functions can be used to save
# all of the data of the experiments.
# But because I already train many experiments
# and only save the model state dicts
# I couldn't retrain all of them again.😊
# ======================================
def run_experiment_save_checkpoint(model,train_loader,val_loader,optimizer,device,epochs=5,
    checkpoint_path="best_model.pt",
    scheduler=None,
    seed=42,
    class_to_idx=None,
    train_transform=None,
    eval_transform=None,
    threshold=None,
    experiment_config=None,
    split_info=None,
    criterion=None,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    best_val_acc = -float("inf")

    for epoch in range(epochs):
        train_metrics = run_one_epoch( model, loader=train_loader,optimizer=optimizer,device=device)
        val_metrics = run_one_epoch(model,loader=val_loader,optimizer=None,device=device)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["Accuracy"])
        history["val_accuracy"].append(val_metrics["Accuracy"])
        history["learning_rate"].append(current_lr)

        if val_metrics["Accuracy"] > best_val_acc:
            best_val_acc = val_metrics["Accuracy"]

            save_experiment_checkpoint(
                path=f"../results/saved/{checkpoint_path}",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_metric=best_val_acc,
                best_epoch=epoch + 1,
                seed=seed,
                class_to_idx=class_to_idx,
                train_transform=train_transform,
                eval_transform=eval_transform,
                threshold=threshold,
                experiment_config=experiment_config,
                split_info=split_info,
                criterion=criterion if criterion is not None else globals().get("criterion"),
                device=device,
            )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train loss: {train_metrics['loss']:.3f} | "
            f"val loss: {val_metrics['loss']:.3f} | "
            f"train acc: {train_metrics['Accuracy']:.1%} | "
            f"val acc: {val_metrics['Accuracy']:.1%} | "
            f"current lr: {current_lr}"
        )

        if scheduler is not None:
            if isinstance(
                    scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_metrics["loss"])

            else:
                scheduler.step()

        if (epoch + 1) % 10 == 0 and (epoch + 1) < epochs:
            for remaining in range(150, 0, -1):
                print(f"\rCooling down... {remaining:03d} seconds remaining", end="")
                time.sleep(1)
            print("\n")

    return history

def serialize_transform(transform):
    return {
        "class": transform.__class__.__name__,
        "repr": repr(transform),
    }


def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "architecture": repr(model),
    }


def get_optimizer_info(optimizer):
    param_groups = []
    for group in optimizer.param_groups:
        group_info = {
            "lr": group.get("lr"),
            "weight_decay": group.get("weight_decay", 0.0),
        }
        if "betas" in group:
            group_info["betas"] = group["betas"]
        if "eps" in group:
            group_info["eps"] = group["eps"]
        if "momentum" in group:
            group_info["momentum"] = group["momentum"]
        param_groups.append(group_info)
    return {
        "name": optimizer.__class__.__name__,
        "param_groups": param_groups,
    }


def get_scheduler_info(scheduler):
    if scheduler is None:
        return None
    return {
        "name": scheduler.__class__.__name__,
        "state": scheduler.state_dict(),
    }


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def save_experiment_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    history=None,
    best_metric=None,
    best_epoch=None,
    seed=42,
    class_to_idx=None,
    train_transform=None,
    eval_transform=None,
    threshold=None,
    experiment_config=None,
    split_info=None,
    criterion=None,
    device=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if class_to_idx is not None:
        class_to_idx = dict(class_to_idx)
        idx_to_class = {
            idx: cls
            for cls, idx in class_to_idx.items()
        }
    else:
        idx_to_class = None

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_info": get_model_info(model),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "optimizer_info": (
            get_optimizer_info(optimizer)
            if optimizer is not None
            else None
        ),
        "scheduler_info": get_scheduler_info(scheduler),
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "split_info": split_info,
        "transforms": {
            "train": (
                serialize_transform(train_transform)
                if train_transform is not None
                else None
            ),
            "eval": (
                serialize_transform(eval_transform)
                if eval_transform is not None
                else None
            ),
        },
        "threshold": threshold,
        "criterion": (
            criterion.__class__.__name__
            if criterion is not None
            else None
        ),
        "experiment_config": experiment_config,
        "seed": seed,
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "git_commit": get_git_commit(),
        },
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")
# =================================
# END
# =================================
def predict_images_from_folder(
    model,
    image_dir,
    transform,
    classes,
    device,
    review_threshold=0.70,
    batch_size=32,
    output_json=None,
    output_plot=None,
    mean=None,
    std=None,
):
    dataset = UnlabeledImageDataset(image_dir=image_dir,transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()

    json_results = []
    predictions = []
    confidences = []
    scores = []

    with torch.no_grad():

        for images, paths in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_classes = probabilities.max(dim=1)
            for i in range(len(images)):
                pred_idx = int(predicted_classes[i].item())
                conf = float(confidence[i].item())
                probs = probabilities[i].cpu().numpy()
                result = {
                    "file_name": Path(paths[i]).name,
                    "path": paths[i],
                    "predicted_class": classes[pred_idx],
                    "confidence": conf,
                    "probabilities": {
                        classes[j]: float(probs[j])
                        for j in range(len(classes))
                    },
                    "needs_review": conf < review_threshold
                }

                json_results.append(result)
                predictions.append(pred_idx)
                confidences.append(conf)
                scores.append(probs)


    if output_json is not None:

        with open(output_json,"w",encoding="utf-8") as f:
            json.dump(json_results,f,indent=4,ensure_ascii=False)

        print(f"JSON saved to: {output_json}")

    print("=" * 70)
    print(f"Total images: {len(dataset)}")

    for result in json_results:
        print(
            f"File: {result['file_name']} | "
            f"Pred: {result['predicted_class']} | "
            f"Conf: {result['confidence']:.2%} | "
            f"Review: {result['needs_review']}"
        )

    if output_plot is not None:
        num_images = len(dataset)
        cols = 3
        rows = math.ceil(num_images / cols)
        fig, axes = plt.subplots(rows,cols,figsize=(20, rows * 5))
        axes = np.array(axes).reshape(-1)
        if mean is None:
            mean = torch.tensor([0.5, 0.5, 0.5])
        if std is None:
            std = torch.tensor([0.5, 0.5, 0.5])
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)

        for ax, idx in zip(axes, range(num_images)):
            image, image_path = dataset[idx]

            # denormalize
            image = image.cpu() * std + mean
            image = image.clamp(0, 1)
            image = image.permute(1, 2, 0)
            ax.imshow(image)
            ax.set_title(
                f"{Path(image_path).name}\n"
                f"Pred: {classes[predictions[idx]]}\n"
                f"Conf: {confidences[idx]:.2%}"
            )
            ax.axis("off")

        for ax in axes[num_images:]:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            output_plot,
            bbox_inches="tight",
            dpi=300,
            format="png"
        )

        plt.show()

    return json_results


def create_labeled_loader(data_path, transform, batch_size=32):
    dataset = ImageFolder(
        root=data_path,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataset, loader