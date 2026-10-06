from pathlib import Path
import torch
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights
from models import resnet18_model, mobilenet_v3_small_model
from utils import (
    predict_images_from_folder,
    create_labeled_loader,
    evaluate_model,
    calculate_classification_metrics,
    run_one_epoch,
    print_metrics,
    plot_confusion_matrices,
    extract_predictions_and_confidence,
    _save_json,
    _predict_labeled_dataset_with_ood
)

from transforms import train_baseline_transform

from ood_detection import (
    load_ood_statistics,
    # predict_with_ood,
    evaluate_ood_folder
)
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ['ambulance','autobus','kamyun','kamyunet','minibus','savari','taxi','vanet']


# with code below we can predict images without labels.
def predict_resnet18_folder_without_ood(image_dir,review_threshold=0.70):

    model_resnet_ft = resnet18_model(num_classes=8).to(device)
    model_resnet_ft.load_state_dict(torch.load("../results/saved/best_resnet18_ft.pt",map_location=device))

    model_resnet_ft.eval()
    weights = ResNet18_Weights.DEFAULT
    resnet_transform = weights.transforms()
    results = predict_images_from_folder(
        model=model_resnet_ft,
        image_dir=image_dir,
        transform=resnet_transform,
        classes=classes,
        device=device,
        review_threshold=review_threshold,
        batch_size=32,
        output_json="../results/test/resnet18_folder_predictions.json",
        output_plot="../results/test/resnet18_folder_predictions.png",
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    )
    return results


def predict_mobilenet_v3_small_folder_without_ood(image_dir,review_threshold=0.70):
    model_mobilenet = mobilenet_v3_small_model(num_classes=8).to(device)
    model_mobilenet.load_state_dict(
        torch.load("../results/saved/mobilenet_v3_small_model_better_augment.pt",map_location=device)
    )

    model_mobilenet.eval()
    results = predict_images_from_folder(
        model=model_mobilenet,
        image_dir=image_dir,
        transform=train_baseline_transform,
        classes=classes,
        device=device,
        review_threshold=review_threshold,
        batch_size=32,
        output_json="../results/test/mobilenetv3_folder_predictions.json",
        output_plot="../results/test/mobilenetv3_folder_predictions.png",
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    return results



# with code below we can predict images with labels.
def predict_test_images_with_res_ft_without_ood(data_path,review_threshold=0.70,batch_size=32):

    weights = ResNet18_Weights.DEFAULT
    resnet_transform = weights.transforms()
    dataset, loader = create_labeled_loader(data_path=data_path,transform=resnet_transform,batch_size=batch_size)
    model_resnet_ft = resnet18_model(num_classes=8).to(device)

    for param in model_resnet_ft.parameters():
        param.requires_grad = False

    for param in model_resnet_ft.layer4.parameters():
        param.requires_grad = True

    for param in model_resnet_ft.fc.parameters():
        param.requires_grad = True

    model_resnet_ft.load_state_dict(
        torch.load("../results/saved/best_resnet18_ft.pt",map_location=device)
    )

    model_resnet_ft.eval()
    result = run_one_epoch(model=model_resnet_ft,loader=loader,optimizer=None,device=device)
    print(result)

    y_true_res_ft, y_pred_res_ft = evaluate_model(
        model_resnet_ft,
        loader,
        device=device
    )

    res_ft_metrics = calculate_classification_metrics(y_true_res_ft,y_pred_res_ft)
    print_metrics(res_ft_metrics,classes)
    print("=" * 70)

    plot_confusion_matrices(
        y_true_res_ft,
        y_pred_res_ft,
        classes=classes,
        experiment_name="resnet18_ft",
        output_dir="../results/test"
    )
    targets, predictions, confidences, scores = (
        extract_predictions_and_confidence(
            model=model_resnet_ft,
            loader=loader,
            device=device,
            loss_type="ce"
        )
    )

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
            "path": dataset.samples[i][0],
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "needs_review": confidence < review_threshold
        })
    json_path = "../results/test/resnet18_ft_predictions_labeled_folder.json"

    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(json_results,f,indent=4,ensure_ascii=False)
    print(f"JSON saved to: {json_path}")

    misclassified_indices = [
        i
        for i, (true_label, pred_label)
        in enumerate(zip(targets, predictions))
        if true_label != pred_label
    ]
    print(f"Total misclassified: {len(misclassified_indices)}")
    for idx in misclassified_indices:

        print(
            f"Index: {idx} | "
            f"True: {classes[targets[idx]]} | "
            f"Pred: {classes[predictions[idx]]} | "
            f"Confidence: {confidences[idx]:.2%} | "
            f"Path: {dataset.samples[idx][0]}"
        )

        print("=" * 50)

    num_images = len(misclassified_indices)

    if num_images == 0:
        print("No misclassified images found.")
        return

    mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(3, 1, 1)

    cols = 3
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(20, rows * 5)
    )

    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(
        axes,
        misclassified_indices
    ):

        image, _ = dataset[idx]
        image = image * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(
            f"True: {classes[targets[idx]]}\n"
            f"Pred: {classes[predictions[idx]]}\n"
            f"Conf: {confidences[idx]:.2%}"
        )
        ax.axis("off")
    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "../results/test/resnet_ft_all_misclassified_labeled_folder.png",
        bbox_inches="tight",
        dpi=300,
        format="png"
    )
    plt.show()


def predict_test_images_MobileNet3_without_ood(data_path,review_threshold=0.70,batch_size=32):
    transform = train_baseline_transform
    dataset, loader = create_labeled_loader(
        data_path=data_path,
        transform=transform,
        batch_size=batch_size
    )
    model_mobilenet_augment = (mobilenet_v3_small_model(num_classes=8).to(device))

    model_mobilenet_augment.load_state_dict(
        torch.load("../results/saved/mobilenet_v3_small_model_better_augment.pt",map_location=device)
    )
    model_mobilenet_augment.eval()
    metrics = run_one_epoch(
        model_mobilenet_augment,
        loader,
        optimizer=None,
        device=device
    )

    print(f"MobileNet metrics: {metrics}")

    y_true, y_pred = evaluate_model(
        model_mobilenet_augment,
        loader,
        device=device
    )

    mobile_metrics = calculate_classification_metrics(y_true,y_pred)
    print_metrics(mobile_metrics,classes)

    plot_confusion_matrices(
        y_true,
        y_pred,
        classes=classes,
        experiment_name="mobilenet_v3_small_model_augment",
        output_dir="../results/test"
    )

    targets, predictions, confidences, scores = (
        extract_predictions_and_confidence(
            model=model_mobilenet_augment,
            loader=loader,
            device=device,
            loss_type="ce"
        )
    )
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
            "path": dataset.samples[i][0],
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "needs_review": (
                confidence < review_threshold
            )
        })

    json_path = "../results/test/mobilenetv3_predictions_labeled_folder.json"

    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(json_results,f,indent=4,ensure_ascii=False)

    print(f"JSON saved to: {json_path}")

    misclassified_indices = [
        i
        for i, (true_label, pred_label)
        in enumerate(zip(targets, predictions))
        if true_label != pred_label
    ]

    print(f"Total misclassified: {len(misclassified_indices)}")

    for idx in misclassified_indices:
        print(
            f"Index: {idx} | "
            f"True: {classes[targets[idx]]} | "
            f"Pred: {classes[predictions[idx]]} | "
            f"Confidence: {confidences[idx]:.2%} | "
            f"Path: {dataset.samples[idx][0]}"
        )

    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    num_images = len(misclassified_indices)

    cols = 3
    rows = math.ceil(num_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))

    axes = np.array(axes).reshape(-1)
    for ax, idx in zip(axes, misclassified_indices):
        image, _ = dataset[idx]

        image = image * std + mean
        image = image.clamp(0, 1)
        image = image.permute(1, 2, 0)
        ax.imshow(image)
        ax.set_title(
            f"True: {dataset.classes[targets[idx]]}\n"
            f"Pred: {dataset.classes[predictions[idx]]}\n"
            f"Conf: {confidences[idx]:.2%}"
        )

        ax.axis("off")

    for ax in axes[num_images:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "../results/test/model_mobilenet_augment_all_misclassified.png",
        bbox_inches="tight",
        dpi=300,
        format="png",
    )

    plt.show()





# ===========
# WITH OOD
# ===========
def predict_resnet18_labeled_folder(data_path,review_threshold=0.85,
    batch_size=32, output_json="../results/test/resnet18_labeled_predictions_with_ood.json"):

    model = resnet18_model(num_classes=8).to(device)
    model.load_state_dict(torch.load("../results/saved/best_resnet18_ft.pt",map_location=device))
    model.eval()

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    dataset, _ = create_labeled_loader(data_path=data_path,transform=transform,batch_size=batch_size)


    ood_stats = load_ood_statistics("../results/saved/resnet_fine-tuned_ood_stats.npz")

    centroids = ood_stats["centroids"]
    cov_inv = ood_stats["cov_inv"]
    threshold = ood_stats["threshold"]

    results = _predict_labeled_dataset_with_ood(
        model=model,
        dataset=dataset,
        centroids=centroids,
        cov_inv=cov_inv,
        threshold=threshold,
        classes=classes,
        device=device,
        review_threshold=review_threshold
    )

    _save_json(results,output_json)

    return results


def predict_resnet18_folder(image_dir, review_threshold=0.85,
    output_json="../results/test/resnet18_folder_predictions_with_ood.json"):

    model = resnet18_model(num_classes=8).to(device)

    model.load_state_dict(torch.load("../results/saved/best_resnet18_ft.pt",map_location=device))
    model.eval()

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    ood_stats = load_ood_statistics("../results/saved/resnet_fine-tuned_ood_stats.npz")

    results = evaluate_ood_folder(
        model=model,
        image_dir=image_dir,
        transform=transform,
        centroids=ood_stats["centroids"],
        cov_inv=ood_stats["cov_inv"],
        threshold=ood_stats["threshold"],
        classes=classes,
        device=device,
        feature_layer_name="avgpool",
        review_threshold=review_threshold
    )
    _save_json(results,output_json)

    return results


# =================================================
# MobilenetV3 small better augmentation prediction
# =================================================
def predict_mobilenet_v3_small_labeled_folder(data_path, review_threshold=0.85, batch_size=32,
    output_json="../results/test/mobilenet_v3_small_labeled_predictions_with_ood.json"):


    model = mobilenet_v3_small_model(num_classes=8).to(device)
    model.load_state_dict(torch.load("../results/saved/mobilenet_v3_small_model_better_augment.pt",map_location=device))

    model.eval()
    transform = train_baseline_transform

    dataset, _ = create_labeled_loader(
        data_path=data_path,
        transform=transform,
        batch_size=batch_size
    )

    ood_stats = load_ood_statistics("../results/saved/mobilenet_v3_small_ood_stats.npz")

    centroids = ood_stats["centroids"]
    cov_inv = ood_stats["cov_inv"]
    threshold = ood_stats["threshold"]

    results = _predict_labeled_dataset_with_ood(
        model=model,
        dataset=dataset,
        centroids=centroids,
        cov_inv=cov_inv,
        threshold=threshold,
        classes=classes,
        device=device,
        review_threshold=review_threshold
    )
    _save_json(results,output_json)

    return results


def predict_mobilenet_v3_small_folder(image_dir,review_threshold=0.85,
    output_json="../results/test/mobilenet_v3_small_folder_predictions_with_ood.json"):

    model = mobilenet_v3_small_model(num_classes=8).to(device)

    model.load_state_dict(torch.load("../results/saved/mobilenet_v3_small_model_better_augment.pt",map_location=device))
    model.eval()
    transform = train_baseline_transform

    ood_stats = load_ood_statistics("../results/saved/mobilenet_v3_small_ood_stats.npz")

    results = evaluate_ood_folder(
        model=model,
        image_dir=image_dir,
        transform=transform,
        centroids=ood_stats["centroids"],
        cov_inv=ood_stats["cov_inv"],
        threshold=ood_stats["threshold"],
        classes=classes,
        device=device,
        feature_layer_name="avgpool",
        review_threshold=review_threshold
    )
    _save_json(results,output_json)

    return results




if __name__ == "__main__":
    # without OOD
    predict_resnet18_folder_without_ood(image_dir="../data/dataset/New folder",review_threshold=0.85)
    predict_mobilenet_v3_small_folder_without_ood(image_dir="../data/dataset/neisan",review_threshold=0.85)

    predict_test_images_with_res_ft_without_ood(data_path="../data/dataset/TEST",review_threshold = 0.85)
    predict_test_images_MobileNet3_without_ood(data_path="../data/dataset/TEST",review_threshold = 0.85)

    # with OOD
    predict_resnet18_labeled_folder(
        data_path="../data/dataset/TEST",
        review_threshold=0.85,
        batch_size=32,
        output_json="../results/test/resnet18_labeled_predictions_with_ood.json"
    )

    predict_resnet18_folder(
        image_dir="../data/dataset/neisan",
        review_threshold= 0.85,
        output_json="../results/test/resnet18_predictions_with_ood.json"
    )

    predict_mobilenet_v3_small_labeled_folder(
        data_path="../data/dataset/TEST",
        review_threshold=0.85,
        batch_size=32,
        output_json="../results/test/mobilenet_v3_small_labeled_predictions_with_ood.json"
    )

    predict_mobilenet_v3_small_folder(
        image_dir="../data/dataset/neisan",
        review_threshold=0.85,
        output_json="../results/test/mobilenet_v3_small_predictions_with_ood.json"
    )




