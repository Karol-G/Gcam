import cv2
import numpy as np
import matplotlib.cm as cm
import PIL
import torch

def save_guided_gcam(filename, gcam, guided_bp):
    guided_gcam = generate_guided_gcam(gcam, guided_bp)
    cv2.imwrite(filename, np.uint8(guided_gcam))

def save_gcam(filename, gcam, raw_image):
    gcam = generate_gcam(gcam, raw_image)
    cv2.imwrite(filename, gcam)

def generate_gcam(gcam, raw_image):
    raw_image = _load_image(raw_image)
    raw_image = _pil2opencv(raw_image)
    raw_image = _tensor2numpy(raw_image)
    gcam = _tensor2numpy(gcam)
    raw_image = _resize_image(raw_image, gcam)
    cmap = cm.jet_r(gcam)[..., :3] * 255.0  # TODO: Still bugged with batch dim
    gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2  # TODO: Still bugged with batch dim
    return np.uint8(gcam)

def generate_guided_gcam(gcam, guided_bp):
    gcam = _tensor2numpy(gcam)
    guided_bp = _tensor2numpy(guided_bp)
    guided_gcam = np.multiply(gcam, guided_bp)
    guided_gcam -= np.min(guided_gcam)
    guided_gcam /= np.max(guided_gcam)
    guided_gcam *= 255.0
    return np.uint8(guided_gcam)

def save_attention_map(filename, attention_map):
    np.save(filename, attention_map)

def save_attention_map_plain(filename, attention_map):
    np.savetxt(filename, attention_map)

def _load_image(image_path):
    if isinstance(image_path, str):
        return cv2.imread(image_path)
    else:
        return image_path

def _pil2opencv(image):
    if isinstance(image, PIL.JpegImagePlugin.JpegImageFile) or isinstance(image, PIL.Image.Image):
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    else:
        return image

def _tensor2numpy(gcam):
    if isinstance(gcam, torch.Tensor):
        gcam = gcam.detach().cpu().numpy()
        if len(gcam.shape) == 4:
            return gcam.transpose(0, 2, 3, 1)
        else:
            return gcam.transpose(1, 2, 0)
    else:
        return gcam

def _resize_image(raw_image, gcam):
    if len(gcam.shape) == 4:
        if raw_image.shape[1:3] == gcam.shape[1:3]:
            return raw_image
        else:
            return cv2.resize(raw_image, gcam.shape[1:3])
    else:
        if raw_image.shape[:2] == gcam.shape[:2]:
            return raw_image
        else:
            return cv2.resize(raw_image, gcam.shape[:2])
