import numpy as np
import cv2
import copy

def preprocessing(attention_map, mask, attention_threshold):
    attention_map = _resize_attention_map(attention_map, mask.shape)
    weights = copy.deepcopy(attention_map)
    mask = np.array(mask, dtype=int)
    attention_map[attention_map < attention_threshold] = 0
    attention_map[attention_map >= attention_threshold] = 1
    attention_map = np.array(attention_map, dtype=int)
    return attention_map, mask, weights

def _resize_attention_map(attention_map, target_shape):
    return cv2.resize(attention_map, tuple(np.flip(target_shape)))

def intersection_over_attention(binary_attention_map, mask, weights):
    intersection = binary_attention_map & mask
    if weights is not None:
        intersection = intersection.astype(np.float) * weights
        binary_attention_map = binary_attention_map.astype(np.float) * weights
    ioa = np.sum(intersection) / np.sum(binary_attention_map)
    return ioa

def intersection_over_union(binary_attention_map, mask, weights):  # TODO: wiou is bad and wrong, maybe not even possible?
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
