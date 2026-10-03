from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import (
    ResNet18_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
)
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from models import resnet18_model
from utils import(
    evaluate_model,
    calculate_classification_metrics,
    run_one_epoch,
    print_metrics,
    plot_confusion_matrices,
    extract_predictions_and_confidence
)
from datasets import test_loader_resnet,test_dataset_resnet,test_loader_base,test_dataset_base





with contextlib.redirect_stdout(io.StringIO()):
    with contextlib.redirect_stderr(io.StringIO()):
        from models import (
            SmallCnn,
            resnet18_model,
            mobilenet_v3_small_model,
            build_mobilenet_v3_large,
            DepthwiseCNN,
        )

DEFAULT_CLASSES = [
    "ambulance",
    "autobus",
    "kamyun",
    "kamyunet",
    "minibus",
    "savari",
    "taxi",
    "vanet",
]


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        requested = torch.device(device)

        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but CUDA is not available.")

        return requested

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
        model_name: str,
        num_classes: int = 8,
        dropout: float = 0.0,
        pooling: str = "max",
) -> nn.Module:
    if model_name == "small_cnn":
        return SmallCnn(
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling,
        )

    if model_name == "resnet18":
        return resnet18_model(num_classes=num_classes)

    if model_name == "mobilenet_v3_small":
        return mobilenet_v3_small_model(num_classes=num_classes)

    if model_name == "mobilenet_v3_large":
        return build_mobilenet_v3_large(num_classes=num_classes)

    if model_name == "depthwise_cnn":
        return DepthwiseCNN(
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        "Use one of: small_cnn, resnet18, mobilenet_v3_small, "
        "mobilenet_v3_large, depthwise_cnn."
    )


def get_inference_transform(
        transform_mode: str,
):
    if transform_mode == "baseline":
        return T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )

    if transform_mode == "resnet":
        return ResNet18_Weights.DEFAULT.transforms()

    raise ValueError("transform_mode must be 'baseline' or 'resnet'.")


def load_state_dict_compatible(
        checkpoint_path: str | Path,
        device: torch.device,
) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        obj = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:

        obj = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]

    if not isinstance(obj, dict):
        raise TypeError(
            "Checkpoint is neither a state_dict nor a checkpoint dict "
            "containing 'model_state_dict'."
        )

    return obj


def load_model(
        checkpoint_path: str | Path,
        model_name: str,
        device: torch.device,
        num_classes: int = 8,
        dropout: float = 0.0,
        pooling: str = "max",
) -> nn.Module:
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
    )

    state_dict = load_state_dict_compatible(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load checkpoint into the requested architecture. "
            "Check model_name, and for SmallCNN check dropout/pooling."
        ) from exc

    model = model.to(device)
    model.eval()

    return model


def preprocess_image(
        image_path: str | Path,
        transform,
) -> torch.Tensor:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Could not open image: {image_path}"
        ) from exc

    tensor = transform(image).unsqueeze(0)
    return tensor


def predict(
        image_path: str | Path,
        checkpoint_path: str | Path,
        model_name: str,
        transform_mode: str,
        loss_type: str = "ce",
        threshold: float = 0.70,
        classes: list[str] | None = None,
        top_k: int = 3,
        dropout: float = 0.0,
        pooling: str = "max",
        device: str | None = None,
) -> dict[str, Any]:
    if loss_type not in {"ce", "bce"}:
        raise ValueError("loss_type must be 'ce' or 'bce'.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    classes = DEFAULT_CLASSES if classes is None else list(classes)

    if len(classes) != 8:
        raise ValueError("This project currently expects exactly 8 classes.")

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    top_k = min(top_k, len(classes))

    inference_device = get_device(device)

    model = load_model(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=inference_device,
        num_classes=len(classes),
        dropout=dropout,
        pooling=pooling,
    )

    transform = get_inference_transform(
        transform_mode=transform_mode,
    )

    image_tensor = preprocess_image(
        image_path=image_path,
        transform=transform,
    ).to(inference_device)

    with torch.inference_mode():
        logits = model(image_tensor)

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(
                f"Expected logits shape [1, num_classes], got {tuple(logits.shape)}"
            )

        logits = logits[0]

        if loss_type == "ce":
            scores = torch.softmax(logits, dim=0)
        else:
            scores = torch.sigmoid(logits)

        predicted_index = int(torch.argmax(logits).item())
        confidence = float(scores[predicted_index].item())

    top_scores, top_indices = torch.topk(scores, k=top_k)

    top_predictions = [
        {
            "class": classes[int(index)],
            "class_index": int(index),
            "score": float(score),
        }
        for score, index in zip(top_scores.cpu(), top_indices.cpu())
    ]

    probability_dict = {
        classes[index]: float(scores[index].item())
        for index in range(len(classes))
    }

    result = {
        "image": str(Path(image_path)),
        "checkpoint": str(Path(checkpoint_path)),
        "model": model_name,
        "loss_type": loss_type,
        "transform": transform_mode,
        "device": str(inference_device),
        "predicted_class": classes[predicted_index],
        "predicted_class_index": predicted_index,
        "confidence": confidence,
        "probabilities": probability_dict,
        "top_predictions": top_predictions,
        "threshold": threshold,
        "needs_review": confidence < threshold,
    }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traffic vehicle classification inference."
    )

    parser.add_argument("image", type=str, help="Path to input image.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to trained .pt checkpoint.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "small_cnn",
            "resnet18",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "depthwise_cnn",
        ],
    )
    parser.add_argument(
        "--transform",
        required=True,
        choices=["baseline", "resnet"],
        help=(
            "Inference preprocessing matching the CURRENT training pipeline. "
            "'resnet' means ResNet18_Weights.DEFAULT.transforms()."
        ),
    )
    parser.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "bce"],
        help="How to convert model logits to class scores.",
    )
    parser.add_argument(
        "--threshold",
        default=0.70,
        type=float,
        help="Confidence threshold for needs_review.",
    )
    parser.add_argument(
        "--top-k",
        default=3,
        type=int,
        help="Number of top predictions to return.",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="SmallCNN/DepthwiseCNN dropout used when the checkpoint was trained.",
    )
    parser.add_argument(
        "--pooling",
        default="max",
        choices=["max", "avg"],
        help="SmallCNN pooling used when the checkpoint was trained.",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Example: cpu, cuda, cuda:0. Default: auto.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        transform_mode=args.transform,
        loss_type=args.loss,
        threshold=args.threshold,
        top_k=args.top_k,
        dropout=args.dropout,
        pooling=args.pooling,
        device=args.device,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


# =======================================
# we can use this part of the code
# because of uncertainty I have
# about above codes.
# =======================================
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ['ambulance','autobus','kamyun','kamyunet','minibus','savari','taxi','vanet']

def predict_test_images_with_res_ft(review_threshold=0.70):
    model_resnet_ft = resnet18_model(num_classes=8).to(device)
    for param in model_resnet_ft.parameters():
        param.requires_grad = False

    for param in model_resnet_ft.layer4.parameters():
        param.requires_grad = True

    for param in model_resnet_ft.fc.parameters():
        param.requires_grad = True

    model_resnet_ft.load_state_dict(
        torch.load(
            "../results/saved/best_resnet18_ft.pt",
            map_location=device
        )
    )
    model_resnet_ft.eval()
    y_true_res_ft, y_pred_res_ft = evaluate_model(
        model_resnet_ft,
        test_loader_resnet,
        device=device
    )
    res_ft_metrics = calculate_classification_metrics(
        y_true_res_ft,
        y_pred_res_ft
    )
    # print(res_ft_metrics)

    print(run_one_epoch(
        model=model_resnet_ft,
        loader=test_loader_resnet,
        device=device,
    ))
    print_metrics(res_ft_metrics, classes)
    print("=" * 70)
    cm_res_ft, cm_res_normalized_ft = plot_confusion_matrices(
        y_true_res_ft,
        y_pred_res_ft,
        classes=classes,
        experiment_name="resnet18_ft",
        output_dir="../results/test",
    )
    targets, predictions, confidences, scores = extract_predictions_and_confidence(
        model=model_resnet_ft,
        loader=test_loader_resnet,
        device=device,
        loss_type="ce"
    )

    # -----------------------------
    # Create JSON output
    # -----------------------------
    json_results = []
    for i in range(len(predictions)):
        predicted_class = classes[predictions[i]]
        true_class = classes[targets[i]]
        confidence = float(confidences[i])

        probabilities = {
            classes[j]: float(scores[i][j])
            for j in range(len(classes))
        }

        json_results.append({
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "needs_review": confidence < 0.70
        })

    with open("../results/test/resnet18_ft_predictions.json","w",encoding="utf-8") as f:
        json.dump(json_results,f,indent=4,ensure_ascii=False)
    print(
        "JSON saved to: "
        "../results/test/resnet18_ft_predictions.json"
    )

    misclassified_indices = [
        i for i, (true_label, pred_label) in enumerate(zip(targets, predictions)) if true_label != pred_label
    ]
    print(f"Total misclassified: {len(misclassified_indices)}")
    for idx in misclassified_indices[:]:
        image, label = test_dataset_resnet[idx]
        print(
            f"Index: {idx} | "
            f"True: {test_dataset_resnet.classes[targets[idx]]} | "
            f"Pred: {test_dataset_resnet.classes[predictions[idx]]} | "
            f"Confidence: {confidences[idx]:.2%} | "
            f"Path: {test_dataset_resnet.samples[idx][0]}\n"
        )
        print("=" * 50)

    weights = ResNet18_Weights.DEFAULT
    mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(3, 1, 1)

    num_images = len(misclassified_indices)

    cols = 3
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))

    axes = np.array(axes).reshape(-1)
    for ax, idx in zip(axes, misclassified_indices):
        image, _ = test_dataset_resnet[idx]

        image = image * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)
        ax.imshow(image)
        ax.set_title(
            f"True: {test_dataset_resnet.classes[targets[idx]]}\n"
            f"Pred: {test_dataset_resnet.classes[predictions[idx]]}\n"
            f"Conf: {confidences[idx]:.2%}"
        )

        ax.axis("off")

    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "../results/test/resnet_ft_all_misclassified.png",
        bbox_inches="tight",
        dpi=300,
        format="png",
    )

    plt.show()


def predict_test_images_MobileNet3(review_threshold=0.7):
    model_mobilenet_augment = mobilenet_v3_small_model(num_classes=8,).to(device)
    model_mobilenet_augment.load_state_dict(
        torch.load(
            "../results/saved/mobilenet_v3_small_model_better_augment.pt",
            map_location=device
        )
    )
    model_mobilenet_augment.eval()
    best_mobile_augment_metrics = run_one_epoch(
        model_mobilenet_augment,
        test_loader_base,
        optimizer=None,
        device=device
    )
    print(f"best_mobile_augment_metrics:{best_mobile_augment_metrics}")

    y_true_mobile_augment, y_pred_mobile_augment = evaluate_model(
        model_mobilenet_augment,
        test_loader_base,
        device=device
    )
    mobile_augment_metrics = calculate_classification_metrics(
        y_true_mobile_augment,
        y_pred_mobile_augment
    )
    print_metrics(mobile_augment_metrics, classes)

    cm_mobile_net_augment, cm_mobile_net_normalized_augment = plot_confusion_matrices(
        y_true_mobile_augment,
        y_pred_mobile_augment,
        classes=classes,
        experiment_name="mobilenet_v3_small_model_augment",
        output_dir="../results/test",
    )
    targets, predictions, confidences, scores = extract_predictions_and_confidence(
        model=model_mobilenet_augment,
        loader=test_loader_base,
        device=device,
        loss_type="ce"
    )

    # -----------------------------
    # Create JSON output
    # -----------------------------
    json_results = []
    for i in range(len(predictions)):
        predicted_class = classes[predictions[i]]
        true_class = classes[targets[i]]
        confidence = float(confidences[i])

        probabilities = {
            classes[j]: float(scores[i][j])
            for j in range(len(classes))
        }

        json_results.append({
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "needs_review": confidence < review_threshold
        })

    with open("../results/test/mobilenetv3_predictions.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=4, ensure_ascii=False)
    print(
        "JSON saved to: "
        "../results/test/mobilenetv3_predictions.json"
    )


    misclassified_indices = [
        i for i, (true_label, pred_label) in enumerate(zip(targets, predictions)) if true_label != pred_label
    ]
    print(f"Total misclassified: {len(misclassified_indices)}")
    for idx in misclassified_indices[:]:
        image, label = test_dataset_base[idx]
        print(
            f"Index: {idx} | "
            f"True: {test_dataset_base.classes[targets[idx]]} | "
            f"Pred: {test_dataset_base.classes[predictions[idx]]} | "
            f"Confidence: {confidences[idx]:.2%} | "
            f"Path: {test_dataset_base.samples[idx][0]}\n"
        )
    mean = torch.tensor([0.5,0.5,0.5]).view(3, 1, 1)
    std = torch.tensor([0.5,0.5,0.5]).view(3, 1, 1)

    num_images = len(misclassified_indices)
    if num_images == 0:
        print("No misclassified images found.")
        return
    cols = 3
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))

    axes = np.array(axes).reshape(-1)
    for ax, idx in zip(axes, misclassified_indices):
        image, _ = test_dataset_resnet[idx]

        image = image * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)
        ax.imshow(image)
        ax.set_title(
            f"True: {test_dataset_resnet.classes[targets[idx]]}\n"
            f"Pred: {test_dataset_resnet.classes[predictions[idx]]}\n"
            f"Conf: {confidences[idx]:.2%}"
        )
        ax.axis("off")
    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "../results/test/model_mobilenet_base_all_misclassified.png",
        bbox_inches="tight",
        dpi=300,
        format="png",
    )

    plt.show()

if __name__ == "__main__":
    # main()
    # predict_test_images_with_res_ft(review_threshold=0.85)
    predict_test_images_MobileNet3(review_threshold=0.85)