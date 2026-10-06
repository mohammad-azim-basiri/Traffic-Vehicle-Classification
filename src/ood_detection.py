import numpy as np
import torch
from pathlib import Path
from PIL import Image

def mahalanobis_min_distance(feat_vec,centroids,cov_inv):

    diffs = feat_vec[None, :] - centroids
    dists = np.einsum("ij,jk,ik->i",diffs,cov_inv,diffs)
    nearest_class = np.argmin(dists)

    return (
        float(dists[nearest_class]),
        int(nearest_class)
    )


@torch.no_grad()
def extract_model_features(model,loader,device,feature_layer_name="avgpool"):

    model.eval()
    all_features = []
    all_labels = []
    batch_features = []

    if not hasattr(model, feature_layer_name):
        raise AttributeError(
            f"Model does not have a layer named "
            f"'{feature_layer_name}'."
        )

    feature_layer = getattr(model,feature_layer_name)


    def hook(module, inputs, output):
        features = output.flatten(1)
        batch_features.append(features.detach().cpu())

    handle = feature_layer.register_forward_hook(hook)

    try:
        for images, labels in loader:
            images = images.to(device)
            model(images)

            if len(batch_features) != 1:
                raise RuntimeError(
                    "Feature hook did not capture "
                    "exactly one batch."
                )

            features = batch_features.pop(0)
            all_features.append(features)
            all_labels.append(labels.cpu())
    finally:
        handle.remove()

    features = torch.cat(all_features,dim=0)
    labels = torch.cat(all_labels,dim=0)

    return (
        features.numpy(),
        labels.numpy()
    )


def calculate_class_centroids(features,labels,num_classes):

    centroids = []

    for class_idx in range(num_classes):

        class_features = features[labels == class_idx]
        if len(class_features) == 0:
            raise ValueError(
                f"No features found for class {class_idx}."
            )

        class_centroid = class_features.mean(axis=0)

        centroids.append(class_centroid)

    return np.stack(centroids,axis=0)


def calculate_shared_covariance(features,labels,centroids,regularization=1e-5):

    num_samples, feature_dim = features.shape
    num_classes = centroids.shape[0]

    residuals = np.empty_like(features)

    for class_idx in range(num_classes):

        class_mask = labels == class_idx

        residuals[class_mask] = (features[class_mask] - centroids[class_idx])

    degrees_of_freedom = (num_samples - num_classes)

    covariance = (residuals.T @ residuals) / degrees_of_freedom


    identity = np.eye(feature_dim)

    covariance_scale = (np.trace(covariance) / feature_dim)

    covariance = covariance + (regularization* covariance_scale* identity)

    cov_inv = np.linalg.inv(covariance)

    return covariance, cov_inv


def calculate_mahalanobis_scores(features,centroids,cov_inv):

    scores = []
    nearest_classes = []

    for feat_vec in features:

        distance, nearest_class = (
            mahalanobis_min_distance(
                feat_vec=feat_vec,
                centroids=centroids,
                cov_inv=cov_inv
            )
        )

        scores.append(distance)
        nearest_classes.append(nearest_class)

    return (
        np.array(scores),
        np.array(nearest_classes)
    )


def calculate_ood_threshold(scores,percentile=95):

    threshold = np.percentile(scores,percentile)

    return float(threshold)


def save_ood_statistics(output_path,centroids,covariance,cov_inv,threshold,percentile=95):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)

    np.savez_compressed(
        output_path,
        centroids=centroids,
        covariance=covariance,
        cov_inv=cov_inv,
        threshold=np.array(threshold),
        percentile=np.array(percentile)
    )

    print(f"OOD statistics saved to: {output_path}")


def load_ood_statistics(input_path):

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"OOD statistics file not found: {input_path}")

    data = np.load(input_path)

    return {
        "centroids": data["centroids"],
        "covariance": data["covariance"],
        "cov_inv": data["cov_inv"],
        "threshold": float(data["threshold"]),
        "percentile": float(data["percentile"])
    }




def predict_with_ood(model,image,centroids,cov_inv,threshold,classes,device,feature_layer_name="avgpool",review_threshold=0.85):

    model.eval()
    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    captured_features = []

    if not hasattr(model,feature_layer_name):
        raise AttributeError(f"Model does not have a layer named '{feature_layer_name}'.")

    feature_layer = getattr(model,feature_layer_name)
    def hook(module, inputs, output):
        features = output.flatten(1)

        captured_features.append(features.detach().cpu())

    handle = feature_layer.register_forward_hook(hook)

    try:
        with torch.no_grad():
            logits = model(image)
    finally:
        handle.remove()

    if len(captured_features) != 1:
        raise RuntimeError("Could not capture model features.")

    features = captured_features[0].numpy()


    probabilities = torch.softmax(logits,dim=1)
    confidence, prediction = torch.max(probabilities,dim=1)

    prediction_idx = int(prediction.item())
    confidence_value = float(confidence.item())

    ood_distance, nearest_class_idx = (
        mahalanobis_min_distance(
            feat_vec=features[0],
            centroids=centroids,
            cov_inv=cov_inv
        )
    )

    is_ood = (ood_distance > threshold)
    needs_review = (confidence_value < review_threshold or is_ood)

    probability_dict = {
        classes[i]: float(
            probabilities[0, i].item()
        )
        for i in range(len(classes))
    }
    return {
        "prediction": classes[prediction_idx],
        "confidence": confidence_value,
        "ood_distance": float(ood_distance),
        "nearest_class": classes[nearest_class_idx],
        "is_ood": bool(is_ood),
        "needs_review": bool(needs_review),
        "probabilities": probability_dict
    }


def evaluate_ood_folder(
        model,image_dir,transform,
        centroids,cov_inv,threshold,
        classes,device,feature_layer_name="avgpool",
        review_threshold=0.85
):


    image_dir = Path(image_dir)

    image_paths = sorted([
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]
    ])

    results = []

    for image_path in image_paths:

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        result = predict_with_ood(
            model=model,
            image=image,
            centroids=centroids,
            cov_inv=cov_inv,
            threshold=threshold,
            classes=classes,
            device=device,
            feature_layer_name=feature_layer_name,
            review_threshold=review_threshold
        )

        result["path"] = str(image_path)
        results.append(result)

    return results








