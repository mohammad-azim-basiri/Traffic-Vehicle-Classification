import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

from torch import optim

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










