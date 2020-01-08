import gc
import torch
from torch.utils.data import DataLoader
import time
import cv2
import matplotlib.cm as cm
import numpy as np
from evaluation_models.grad_cam import grad_cam

#from data.coco_yolo_dataset import CocoYoloDataset as Dataset
#from models.yolo_model import YoloModel as Model
#from data.tumor_seg_dataset import TumorDataset as Dataset
#from models.tumor_seg_model import TumorSegModel as Model

DEVICE = "cuda" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FOBIDDEN: To call torch.no_grad, detach, .long(), .float() ... during forward

OVERLAP_SCORE_THRESHOLD = 0.5
ATTENTION_THRESHOLD = 80
SAVE_BAD_RESULTS = False

def evaluate_dataset(model, dataset):
    #dataset = Dataset(device=DEVICE)
    dataset_len = dataset.__len__()

    #model = Model(device=DEVICE)
    model.eval()
    is_backward_ready = model.is_backward_ready()
    layer = 'auto'
    model_GCAM = grad_cam.GradCAM(model=model)
    #model_GCAM = grad_cam.GradCAM(model=model, candidate_layers=[layer])
    model_GBP = grad_cam.GuidedBackPropagation(model=model)

    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = time.time()
    overlap_percentage = []

    for i, batch in enumerate(data_loader):
        print_progress(start, i, dataset_len, batch_size)
        _ = model_GCAM.forward(batch["img"])
        _ = model_GBP.forward(batch["img"])
        is_ok = model_GCAM.model.get_ok_list()
        # print("is_ok: {}".format(is_ok))
        #cv2.imwrite("gt.png", batch["gt"].squeeze().numpy())

        if True in is_ok: # Only if object are detected
            model_GBP.backward()
            attention_map_GBP = model_GBP.generate()
            model_GCAM.backward()
            attention_map_GCAM = model_GCAM.generate(target_layer=layer, dim=2)

        # TODO: Evaluation machen
        # TODO: Guided-Grad-CAM (Fehlerhaft)
        # TODO: Save images with low overlap score

        #print("attention_map_GCAM: {}".format(attention_map_GCAM))

        for j in range(batch_size):
            if is_ok[j]:
                check_nans(attention_map_GCAM[j], i, j)
                overlap_score = evaluate_image(attention_map_GCAM[j], batch["gt"][j])
                overlap_percentage.append(overlap_score)
                if SAVE_BAD_RESULTS and overlap_score < OVERLAP_SCORE_THRESHOLD:
                    # save_attention_map(filename="/visinf/projects_students/shared_vqa/pythia/attention_maps/gradcam/" + str(annId) + ".npy", attention_map=attention_map_GCAM)
                    # save_attention_map_plain(filename="results/attention_map_" + j + ".txt", attention_map=attention_map_GCAM)
                    save_gradcam(filename="results/gcam/attention_map_" + str(i * batch_size + j) + "_score_" + str(round(overlap_score*100)) + ".png", gcam=attention_map_GCAM[j], filepath=batch["filepath"][j])
                    #save_gradient(filename="results/guided-gcam/attention_map_" + str(i * batch_size + j) + ".png", gradient=torch.mul(attention_map_GCAM[j], attention_map_GBP[j]))

            else:
                print("Warning: No class detected in the {}-th image of batch {}!".format(j, i))
                #save_image(batch["filepath"][j], i * batch_size + j)
                overlap_percentage.append(0.0)

        avg_overlap_percentage = sum(overlap_percentage) / len(overlap_percentage)
        print("avg_overlap_percentage: {}%".format(round(avg_overlap_percentage*100, 2)))

        # if i >= 4:
        #     break

    np.savetxt("results/overlap_percentage.txt", overlap_percentage)
    np.save("results/overlap_percentage", overlap_percentage)
    gc.collect()
    torch.cuda.empty_cache()

def check_nans(attention_map, i, j):
    nans = torch.sum(torch.isnan(attention_map))
    if nans > 0:
        print("Warning: The {}-th attention map of batch {} contains {} NaN values!".format(j, i, nans))

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

def evaluate_image(attention_map, ground_truth): # This evaluation is not suposed to be IoU
    ground_truth = ground_truth.numpy()
    attention_map = attention_map.squeeze(0).squeeze(0)
    attention_map = attention_map.cpu().numpy()
    #attention_map = attention_map.detach().cpu().numpy()
    attention_map -= attention_map.min()
    attention_map /= attention_map.max()
    attention_map *= 255.0
    attention_map = np.array(attention_map, dtype=np.uint8)
    attention_map[attention_map < ATTENTION_THRESHOLD] = 0
    # TODO: Erode or dilate ground truth
    overlap_percentage = cv2.bitwise_and(attention_map, attention_map, mask=ground_truth)
    attention_map = np.sum(attention_map)
    overlap_percentage = np.sum(overlap_percentage) / attention_map
    print("overlap_percentage: {}%".format(round(overlap_percentage*100, 2)))

    return overlap_percentage

def print_layer_names(model):
    print(*list(model.named_modules()), sep='\n')

# if __name__ == "__main__":
#     #print_layer_names()
#     run()