import cv2
import numpy as np
import matplotlib.cm as cm

MIN_SHAPE = (500, 500)

def save_attention_map(filename, attention_map, backend, dim, data=None):
    attention_map = generate_attention_map(attention_map, backend, dim, data)
    cv2.imwrite(filename, attention_map)

def generate_attention_map(attention_map, heatmap, dim, data=None):
    if dim == 2:
        if heatmap:
            return generate_gcam2d(attention_map, data)
        else:
            return generate_guided_bp2d(attention_map)
    elif dim == 3:
        if heatmap:
            return generate_gcam3d(attention_map, data)
        else:
            return generate_guided_bp3d(attention_map)
    else:
        raise RuntimeError("Unsupported dimension. Only 2D and 3D data is supported.")

def generate_gcam2d(attention_map, data=None):
    assert(len(attention_map.shape) == 2)  # No batch dim
    assert(isinstance(attention_map, np.ndarray))  # Not a tensor
    assert(isinstance(data, np.ndarray) or isinstance(data, str) or data is None)  # Not PIL
    assert(data is None or len(data.shape) == 2 or data.shape[2] == 3)  # Format (h,w) or (h,w,c)

    if data is not None:
        data = _load_data(data)
        attention_map = _resize_attention_map(attention_map, data.shape[:2])
        cmap = cm.jet_r(attention_map)[..., :3] * 255.0  # TODO: Still bugged with batch dim
        attention_map = (cmap.astype(np.float) + data.astype(np.float)) / 2  # TODO: Still bugged with batch dim
    else:
        attention_map = _resize_attention_map(attention_map, MIN_SHAPE)
        attention_map = cm.jet_r(attention_map)[..., :3] * 255.0  # TODO: Still bugged with batch dim
    return np.uint8(attention_map)

def generate_guided_bp2d(attention_map):
    assert(len(attention_map.shape) == 2)
    assert (isinstance(attention_map, np.ndarray))  # Not a tensor
    attention_map -= np.min(attention_map)
    attention_map /= np.max(attention_map)
    attention_map *= 255.0
    attention_map = _resize_attention_map(attention_map, MIN_SHAPE)
    return np.uint8(attention_map)

def generate_gcam3d(attention_map, data=None):
    assert(len(attention_map.shape) == 3)  # No batch dim
    assert(isinstance(attention_map, np.ndarray))  # Not a tensor
    assert(isinstance(data, np.ndarray) or data is None)  # Not PIL
    assert(data is None or len(data.shape) == 3)

    if data is not None:
        attention_map = _resize_attention_map(attention_map, data.shape[:3])
        cmap = cm.jet_r(attention_map)[..., :3] * 255.0  # TODO: Still bugged with batch dim
        attention_map = (cmap.astype(np.float) + data.astype(np.float)) / 2  # TODO: Still bugged with batch dim
    else:
        attention_map = _resize_attention_map(attention_map, MIN_SHAPE)
        attention_map = cm.jet_r(attention_map)[..., :3] * 255.0  # TODO: Still bugged with batch dim
    return np.uint8(attention_map)

def generate_guided_bp3d(attention_map):
    assert(len(attention_map.shape) == 2)
    assert (isinstance(attention_map, np.ndarray))  # Not a tensor
    attention_map -= np.min(attention_map)
    attention_map /= np.max(attention_map)
    attention_map *= 255.0
    attention_map = _resize_attention_map(attention_map, MIN_SHAPE)
    return np.uint8(attention_map)

def _load_data(data_path):
    if isinstance(data_path, str):
        return cv2.imread(data_path)
    else:
        return data_path

def _resize_attention_map(attention_map, min_shape):
    attention_map_shape = attention_map.shape[:2]
    if min(min_shape) < min(attention_map_shape):
        attention_map = cv2.resize(attention_map, tuple(np.flip(attention_map_shape)))
    else:
        resize_factor = int(min(min_shape) / min(attention_map_shape))
        data_shape = (attention_map_shape[0] * resize_factor, attention_map_shape[1] * resize_factor)
        attention_map = cv2.resize(attention_map, tuple(np.flip(data_shape)))
    return attention_map

def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
