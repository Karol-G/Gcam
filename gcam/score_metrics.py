import numpy as np
import cv2

def preprocessing(attention_map, mask, attention_threshold, dilate_value):
    mask = _dilate_mask(mask, dilate_value)
    attention_map = _resize_attention_map(attention_map, mask.shape)  # _resize_image(ground_truth, attention_map)
    mask = np.array(mask, dtype=np.uint8)
    attention_map -= np.min(attention_map)
    attention_map /= np.max(attention_map)
    attention_map[attention_map < attention_threshold] = 0
    attention_map[attention_map >= attention_threshold] = 1
    attention_map *= 255.0
    attention_map = np.array(attention_map, dtype=np.uint8)
    return attention_map, mask

def _dilate_mask(mask, dilate_value):
    if dilate_value > 0:
        kernel = np.ones((dilate_value, dilate_value), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)
    return mask

def _resize_attention_map(attention_map, target_shape):
    return cv2.resize(attention_map, tuple(np.flip(target_shape)))

def overlap_score(attention_map, mask):  # TODO: Rename method
    overlap_percentage = cv2.bitwise_and(attention_map, attention_map, mask=mask)
    attention_map = np.sum(attention_map)
    overlap_percentage = np.sum(overlap_percentage) / attention_map
    return overlap_percentage
