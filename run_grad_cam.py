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

DEVICE = "cpu"

# FOBIDDEN: To call torch.no_grad, detach, .long(), .float() ... during forward

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

def run():
    annotation_filepath = '/visinf/projects_students/shared_vqa/mscoco/coco-annotations/instances_train2014.json'
    dataset_path = '/visinf/projects_students/shared_vqa/mscoco/train2014/'
    dataset = Dataset(annotation_filepath, dataset_path, device=DEVICE)
    dataset_len = dataset.__len__()

    model = Model(device=DEVICE)
    model.eval()
    #layer = 'model.module_list.93'
    #layer = 'model.module_list.81'
    #layer = 'model.module_list.105'
    layer = 'auto'
    model_GCAM = grad_cam.GradCAM(model=model)
    #model_GCAM = grad_cam.GradCAM(model=model, candidate_layers=[layer])

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    start = time.time()

    for j, batch in enumerate(data_loader):
        print_progress(start, j, dataset_len)
        probs = model_GCAM.forward(batch["img"])
        #print("probs: {}".format(probs))
        classes = model_GCAM.model.get_classes()

        model_GCAM.backward(ids=0)
        attention_map_GradCAM = model_GCAM.generate(target_layer=layer)

        # TODO: Ground truth für alle class names laden? gt wäre dann dictionary von class gt_image?
        # TODO: Handle batch size (In demo1 auf github wird richtig gemacht)
        # print("attention_map_GradCAM: {}".format(attention_map_GradCAM))
        # save_attention_map(filename="/visinf/projects_students/shared_vqa/pythia/attention_maps/gradcam/" + str(annId) + ".npy", attention_map=attention_map_GradCAM)
        # save_attention_map_plain(filename="results/attention_map_" + j + ".txt", attention_map=attention_map_GradCAM)
        save_gradcam(filename="results/attention_map_" + str(j) + "_" + classes[0] + ".png", gcam=attention_map_GradCAM, filepath=batch["filepath"][0])

        # break

    gc.collect()
    torch.cuda.empty_cache()

def print_progress(start, j, dataset_len):
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
    print("Iteration: {} | Progress: {}% | Finished in: {}d {}h {}m {}s | Time Per Annotation: {}s".format(j, round(
        progress, 6), round(day), round(hour), round(minutes), round(seconds), round(time_per_annotation, 2)))


def print_layer_names():
    print(*list(Model(device="cpu").named_modules()), sep='\n')

if __name__ == "__main__":
    #print_layer_names()
    run()