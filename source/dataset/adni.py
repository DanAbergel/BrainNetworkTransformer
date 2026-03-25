"""
Bridge between FAIR ADNI data and BrainNetworkTransformer.

FAIR format:  parcellated tensor (N, T, R) = (812, 140, 200)
              + index_to_name.json + imageID_to_labels.json

BNT expects:  time_series (N, R, T), pearson (N, R, R), labels (N,)
"""

import json
import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_adni_data(cfg: DictConfig):
    """Load ADNI data from FAIR project format into BNT format.

    Reads the parcellated tensor and label JSONs, computes Pearson
    correlation matrices, and returns tensors ready for BNT.

    Returns: (final_timeseires, final_pearson, labels, labels_np)
        - final_timeseires: (N, 200, 140) z-scored time series
        - final_pearson:    (N, 200, 200) Pearson correlation matrices
        - labels:           (N,) float tensor of binary labels
        - labels_np:        (N,) numpy array for stratified splitting
    """
    # --- Load parcellated tensor (N, T, R) ---
    parcellated = torch.load(cfg.dataset.parcellated_path, weights_only=True)
    # parcellated shape: (812, 140, 200)

    # --- Load labels ---
    with open(cfg.dataset.index_json) as f:
        index_to_name = json.load(f)
    with open(cfg.dataset.labels_json) as f:
        image_labels = json.load(f)

    label_column = cfg.dataset.label_column

    labels = []
    valid_indices = []
    for idx in sorted(index_to_name.keys(), key=int):
        entry = index_to_name[idx]
        image_id = entry["image_id"]
        if image_id in image_labels and label_column in image_labels[image_id]:
            val = image_labels[image_id][label_column]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                labels.append(int(val))
                valid_indices.append(int(idx))

    labels = np.array(labels)
    valid_indices = np.array(valid_indices)

    # Filter parcellated to valid subjects only
    parcellated = parcellated[valid_indices]  # (N_valid, 140, 200)

    # --- Convert to BNT format ---
    # Time series: (N, T, R) -> (N, R, T)
    timeseries_np = parcellated.numpy().transpose(0, 2, 1)  # (N, 200, 140)

    # Pearson correlation: for each subject, correlate the 200 regions
    n_subjects = timeseries_np.shape[0]
    n_regions = timeseries_np.shape[1]
    pearson_np = np.zeros((n_subjects, n_regions, n_regions), dtype=np.float32)
    for i in range(n_subjects):
        # corrcoef on (200, 140) -> (200, 200) correlation matrix
        corr = np.corrcoef(timeseries_np[i])  # (200, 200)
        # Handle NaN (constant regions -> NaN correlation)
        corr = np.nan_to_num(corr, nan=0.0)
        pearson_np[i] = corr

    # Z-score time series
    scaler = StandardScaler(mean=np.mean(timeseries_np),
                            std=np.std(timeseries_np))
    timeseries_np = scaler.transform(timeseries_np)

    # Convert to tensors
    final_timeseires = torch.from_numpy(timeseries_np).float()
    final_pearson = torch.from_numpy(pearson_np).float()
    labels_tensor = torch.from_numpy(labels).float()

    # Set dataset config
    with open_dict(cfg):
        cfg.dataset.node_sz = n_regions          # 200
        cfg.dataset.node_feature_sz = n_regions  # 200
        cfg.dataset.timeseries_sz = parcellated.shape[1]  # 140

    print(f"ADNI loaded: {n_subjects} subjects, {n_regions} regions, "
          f"label={label_column}, classes={np.unique(labels).tolist()}")

    return final_timeseires, final_pearson, labels_tensor, labels
