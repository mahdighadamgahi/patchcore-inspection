"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import csv
import os

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

def save_image_labels(
    anomaly_prediction_weights, sample_names, results_path, lower_threshold=.1, upper_threshold=.4
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR) and saves the names of test samples in a CSV file with three labels.

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
        sample_names: [list] [N] Names of the samples.
        output_csv_path: [str] Path to save the output CSV file.
        lower_threshold: [float] Lower threshold for determining ambiguous label.
        upper_threshold: [float] Upper threshold for determining anomaly label.
    """
    
    # Assign labels based on the thresholds
    labels = []
    for weight in anomaly_prediction_weights:
        if weight < lower_threshold:
            labels.append(0)  # Normal
        elif weight >= upper_threshold:
            labels.append(1)  # Anomaly
        else:
            labels.append(2)  # Ambiguous
    savename = os.path.join(results_path, "labels.csv")
    # Save the names and labels of test samples in a CSV file
    with open(savename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Sample Name', 'Predicted Label'])
        for name, label in zip(sample_names, labels):
            csv_writer.writerow([name, label])

    return { "threshold": lower_threshold,"threshold_2": upper_threshold}


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
