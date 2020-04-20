import numpy as np
import torch
import cv2
import copy

def comp_score(attention_map, mask, metric="wioa", threshold=0.3):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)
    allowed = [0, 1, 0.0, 1.0]
    if np.min(mask) in allowed and np.max(mask) in allowed:
        mask = mask.astype(int)
    else:
        raise TypeError("Mask values need to be 0/1")
    binary_attention_map, mask, weights = _preprocessing(attention_map, mask, threshold)
    if metric[0] != "w":
        weights = None
    if metric == "ioa" or metric == "wioa":
        score = _intersection_over_attention(binary_attention_map, mask, weights)
    elif metric == "iou" or metric == "wiou":
        score = _intersection_over_union(binary_attention_map, mask, weights)
    elif callable(metric):
        score = metric(attention_map, mask, attention_map, weights)
    else:
        raise AttributeError("Metric does not exist")
    return score

def _preprocessing(attention_map, mask, attention_threshold):
    attention_map = _resize_attention_map(attention_map, mask.shape)
    weights = copy.deepcopy(attention_map)
    mask = np.array(mask, dtype=int)
    attention_map[attention_map < attention_threshold] = 0
    attention_map[attention_map >= attention_threshold] = 1
    attention_map = np.array(attention_map, dtype=int)
    return attention_map, mask, weights

def _resize_attention_map(attention_map, target_shape):
    return cv2.resize(attention_map, tuple(np.flip(target_shape)))

def _intersection_over_attention(binary_attention_map, mask, weights):
    intersection = binary_attention_map & mask
    if weights is not None:
        intersection = intersection.astype(np.float) * weights
        binary_attention_map = binary_attention_map.astype(np.float) * weights
    ioa = np.sum(intersection) / np.sum(binary_attention_map)
    return ioa

def _intersection_over_union(binary_attention_map, mask, weights):  # TODO: wiou is bad and wrong, maybe not even possible?
    intersection = binary_attention_map & mask
    if weights is not None:
        outer_attention = binary_attention_map - intersection
        outer_attention = outer_attention.astype(np.float) * weights
        union = outer_attention + mask.astype(np.float)
        intersection = intersection.astype(np.float) * weights
    else:
        union = binary_attention_map | mask
    iou = np.sum(intersection) / np.sum(union).astype(np.float)
    return iou
