import numpy as np
import cv2

def overlap_score(attention_map, ground_truth, attention_threshold, dilate):  # This evaluation is not suposed to be IoU  # TODO: Rename method
    ground_truth = _dilate_mask(ground_truth, dilate)
    attention_map = _resize_attention_map(attention_map, ground_truth.shape)  # _resize_image(ground_truth, attention_map)
    ground_truth = np.array(ground_truth, dtype=np.uint8)
    attention_map -= np.min(attention_map)  # attention_map.min()
    attention_map /= np.max(attention_map)  # attention_map.max()
    attention_map *= 255.0
    attention_map = np.array(attention_map, dtype=np.uint8)
    attention_map[attention_map < attention_threshold] = 0
    # TODO: Non-linear weighting?
    overlap_percentage = cv2.bitwise_and(attention_map, attention_map, mask=ground_truth)
    attention_map = np.sum(attention_map)
    overlap_percentage = np.sum(overlap_percentage) / attention_map
    return overlap_percentage

def _dilate_mask(mask, dilate):
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)
    return mask

def _resize_attention_map(attention_map, target_shape):
    return cv2.resize(attention_map, tuple(np.flip(target_shape)))