import cv2
import numpy as np
import matplotlib.cm as cm
import PIL

def save_guided_gcam(filename, gcam, guided_bp):
    guided_gcam = np.multiply(gcam, guided_bp)
    guided_gcam = guided_gcam.transpose(1, 2, 0)
    guided_gcam -= np.min(guided_gcam)
    guided_gcam /= np.max(guided_gcam)
    guided_gcam *= 255.0
    cv2.imwrite(filename, np.uint8(guided_gcam))

def save_gcam(filename, gcam, raw_image, paper_cmap=False):
    raw_image = _load_image(raw_image)
    raw_image = _pil2opencv(raw_image)
    #attention_map = _gcam2numpy(attention_map)
    raw_image = _resize_image(raw_image, gcam)
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

def save_attention_map(filename, attention_map):
    #attention_map = attention_map.squeeze().cpu().numpy()
    np.save(filename, attention_map)

def save_attention_map_plain(filename, attention_map):
    #attention_map = attention_map.squeeze().cpu().numpy()
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

def _gcam2numpy(gcam):
    if not (type(gcam) is np.ndarray):
        return gcam.squeeze().cpu().numpy()
    else:
        return gcam

def _resize_image(raw_image, gcam):
    if not (raw_image.shape[:2] == gcam.shape[:2]):
        return cv2.resize(raw_image, gcam.shape[:2])
    else:
        return raw_image
