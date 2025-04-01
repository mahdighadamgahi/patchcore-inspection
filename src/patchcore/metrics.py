"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels, sample_names
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR) and prints the names of failed test samples.

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
        sample_names: [list] [N] Names of the samples.
    """
    y_val_truth, y_test_truth, y_val_pred, y_test_pred, val_names, test_names = train_test_split(
        anomaly_ground_truth_labels, anomaly_prediction_weights, sample_names, test_size=0.5, random_state=42
    )

    fpr, tpr, thresholds = metrics.roc_curve(y_val_truth, y_val_pred)
    J = tpr - fpr
    optimal_threshold = thresholds[np.argmax(J)]
    y_pred = (y_test_pred >= optimal_threshold).astype(int)
    accuracy = metrics.accuracy_score(y_test_truth, y_pred)

    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)



    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": optimal_threshold, "accuracy": accuracy}

def save_imagewise_retrieval_labels(
    anomaly_prediction_weights, anomaly_ground_truth_labels, sample_names
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR) and prints the names of failed test samples.

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
        sample_names: [list] [N] Names of the samples.
    """
    y_val_truth, y_test_truth, y_val_pred, y_test_pred, val_names, test_names = train_test_split(
        anomaly_ground_truth_labels, anomaly_prediction_weights, sample_names, test_size=0.5, random_state=42
    )

    fpr, tpr, thresholds = metrics.roc_curve(y_val_truth, y_val_pred)
    J = tpr - fpr
    optimal_threshold = thresholds[np.argmax(J)]
    y_pred = (y_test_pred >= optimal_threshold).astype(int)
    accuracy = metrics.accuracy_score(y_test_truth, y_pred)

    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)

    # Identify and print the names of failed test samples
    failed_samples = [test_names[i] for i in range(len(y_test_truth)) if y_test_truth[i] != y_pred[i]]
    print("Failed test samples:", failed_samples)

    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": optimal_threshold, "accuracy": accuracy}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }
