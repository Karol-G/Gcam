import cv2
import numpy as np
import matplotlib.cm as cm
import PIL
import torch

MIN_SHAPE = (500, 500)

def save_guided_gcam(filename, gcam, guided_bp):
    guided_gcam = generate_guided_gcam(gcam, guided_bp)
    cv2.imwrite(filename, guided_gcam)

def save_gcam(filename, gcam, image=None, min_shape=True):
    gcam = generate_gcam(gcam, image=image)
    cv2.imwrite(filename, gcam)

def generate_gcam(gcam, image=None, min_shape=True):
    assert(len(gcam.shape) == 2)  # No batch dim
    assert(isinstance(gcam, np.ndarray))  # Not a tensor
    assert(isinstance(image, np.ndarray) or isinstance(image, str) or image is None)  # Not PIL
    assert(image is None or len(image.shape) == 2 or image.shape[2] == 3)  # Format (h,w) or (h,w,c)

    if image is not None:
        image = _load_image(image)
        #raw_image = _pil2opencv(raw_image)
        #raw_image = _tensor2numpy(raw_image)
        #image = _resize_image(image, gcam)
        gcam = _resize_gcam(gcam, image.shape[:2])
        #gcam = _tensor2numpy(gcam)
        cmap = cm.jet_r(gcam)[..., :3] * 255.0  # TODO: Still bugged with batch dim
        gcam = (cmap.astype(np.float) + image.astype(np.float)) / 2  # TODO: Still bugged with batch dim
    else:
        #gcam = _tensor2numpy(gcam)
        gcam_shape = gcam.shape[:2]
        if min(MIN_SHAPE) < min(gcam_shape):
            gcam = _resize_gcam(gcam, gcam_shape)
        else:
            resize_factor = int(min(MIN_SHAPE) / min(gcam_shape))
            gcam_shape = (gcam_shape[0] * resize_factor, gcam_shape[1] * resize_factor)
            gcam = _resize_gcam(gcam, gcam_shape)
        gcam = cm.jet_r(gcam)[..., :3] * 255.0  # TODO: Still bugged with batch dim
    return np.uint8(gcam)

def generate_guided_gcam(gcam, guided_bp):
    assert(len(gcam.shape) == 2)
    assert (len(guided_bp.shape) == 2)
    assert(isinstance(gcam, np.ndarray))
    assert(isinstance(guided_bp, np.ndarray))
    #gcam = _tensor2numpy(gcam)
    #guided_bp = _tensor2numpy(guided_bp)
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

# def _pil2opencv(image):
#     if isinstance(image, PIL.JpegImagePlugin.JpegImageFile) or isinstance(image, PIL.Image.Image):
#         return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#     else:
#         return image
#
# def _tensor2numpy(gcam):
#     if isinstance(gcam, torch.Tensor):
#         gcam = gcam.detach().cpu().numpy()
#         if len(gcam.shape) == 4:
#             return gcam.transpose(0, 2, 3, 1)
#         else:
#             return gcam.transpose(1, 2, 0)
#     else:
#         return gcam

def _resize_gcam(gcam, image_shape):
    if image_shape == gcam.shape[:2]:
        return gcam
    else:
        return cv2.resize(gcam, tuple(np.flip(image_shape)))

# def _resize_image(image, gcam):
#     if len(gcam.shape) == 4:
#         if image.shape[1:3] == gcam.shape[1:3]:
#             return image
#         else:
#             return cv2.resize(image, gcam.shape[1:3])
#     else:
#         if image.shape[:2] == gcam.shape[:2]:
#             return image
#         else:
#             return cv2.resize(image, gcam.shape[:2])
