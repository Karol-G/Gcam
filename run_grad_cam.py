import gc
import torch
from torch.utils.data import DataLoader
import time
import cv2
import matplotlib.cm as cm
import numpy as np
from evaluation_models.grad_cam import grad_cam

from data.coco_yolo_dataset import CocoYoloDataset as Dataset
from models.yolo_model import YoloModel as Model
#from data.tumor_seg_dataset import TumorDataset as Dataset
#from models.tumor_seg_model import TumorSegModel as Model

DEVICE = "cuda" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FOBIDDEN: To call torch.no_grad, detach, .long(), .float() ... during forward

def run():
    dataset = Dataset(device=DEVICE)
    dataset_len = dataset.__len__()

    model = Model(device=DEVICE)
    model.eval()
    is_backward_ready = model.is_backward_ready()
    layer = 'auto'
    model_GCAM = grad_cam.GradCAM(model=model)
    #model_GCAM = grad_cam.GradCAM(model=model, candidate_layers=[layer])
    model_GBP = grad_cam.GuidedBackPropagation(model=model)

    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = time.time()

    for i, batch in enumerate(data_loader):
        print_progress(start, i, dataset_len, batch_size)
        _ = model_GCAM.forward(batch["img"])
        _ = model_GBP.forward(batch["img"])
        classes = model_GCAM.model.get_classes()

        if classes[0]: # Only if object are detected
            model_GBP.backward(ids=0, is_backward_ready=is_backward_ready)
            attention_map_GBP = model_GBP.generate()
            model_GCAM.backward(ids=0, is_backward_ready=is_backward_ready)
            attention_map_GCAM = model_GCAM.generate(target_layer=layer, dim=2)

        # TODO: Guided-Grad-CAM (Fehlerhaft)
        # TODO: Evaluation machen
        # TODO: Ground truth für alle class names laden? gt wäre dann dictionary von class gt_image?

        #print("attention_map_GCAM: {}".format(attention_map_GCAM))

        for j in range(batch_size):
            if classes[j]:
                nans = torch.sum(torch.isnan(attention_map_GCAM[j]))
                if nans > 0:
                    print("Warning: The {}-th attention map of batch {} contains {} NaN values!".format(j, i, nans))
                # save_attention_map(filename="/visinf/projects_students/shared_vqa/pythia/attention_maps/gradcam/" + str(annId) + ".npy", attention_map=attention_map_GCAM)
                # save_attention_map_plain(filename="results/attention_map_" + j + ".txt", attention_map=attention_map_GCAM)
                save_gradcam(filename="results/gcam/attention_map_" + str(i * batch_size + j) + "_" + classes[j][0] + ".png", gcam=attention_map_GCAM[j], filepath=batch["filepath"][j])
                save_gradient(filename="results/guided-gcam/attention_map_" + str(i * batch_size + j) + "_" + classes[j][0] + ".png", gradient=torch.mul(attention_map_GCAM[j], attention_map_GBP[j]))
            else:
                save_image(batch["filepath"][j], i * batch_size + j)

        break

    gc.collect()
    torch.cuda.empty_cache()

def print_progress(start, j, dataset_len, batch_size):
    j = j * batch_size
    progress = ((j + 1) / dataset_len) * 100
    elapsed = time.time() - start
    time_per_annotation = elapsed / (j + 1)

    finished_in = time_per_annotation * (dataset_len - (j + 1))
    day = finished_in // (24 * 3600)
    finished_in = finished_in % (24 * 3600)
    hour = finished_in // 3600
    finished_in %= 3600
    minutes = finished_in // 60
    finished_in %= 60
    seconds = finished_in
    print("Iteration: {} | Progress: {}% | Finished in: {}d {}h {}m {}s | Time Per Batch: {}s".format(j, round(
        progress, 6), round(day), round(hour), round(minutes), round(seconds), round(time_per_annotation, 2)))

def save_image(filepath, index):
    raw_image = cv2.imread(filepath)
    cv2.imwrite("results/attention_map_" + str(index) + "_None.png", raw_image)

def save_gradient(filename, gradient):
    gradient = gradient.squeeze(0)
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))

def save_gradcam(filename, gcam, filepath, paper_cmap=False):
    gcam = gcam.squeeze(0).squeeze(0)
    gcam = gcam.cpu().numpy()
    raw_image = cv2.imread(filepath)
    raw_image = cv2.resize(raw_image, gcam.shape)
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

def save_attention_map(filename, attention_map):
    attention_map = attention_map.squeeze(0).squeeze(0)
    attention_map = attention_map.cpu().numpy()
    np.save(filename, attention_map)

def save_attention_map_plain(filename, attention_map):
    attention_map = attention_map.squeeze(0).squeeze(0)
    attention_map = attention_map.cpu().numpy()
    np.savetxt(filename, attention_map)

def print_layer_names():
    print(*list(Model(device="cpu").named_modules()), sep='\n')

if __name__ == "__main__":
    #print_layer_names()
    run()