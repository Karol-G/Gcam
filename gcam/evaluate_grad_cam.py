import gc
import torch
from torch.utils.data import DataLoader
import time
import cv2
import numpy as np
from gcam.grad_cam import grad_cam
from gcam.grad_cam.gradcam_utils import *
from pathlib import Path
from collections import defaultdict
import pickle

# FOBIDDEN: To call torch.no_grad, detach, .long(), .float() ... during forward

OVERLAP_SCORE_THRESHOLD = 0.5
ATTENTION_THRESHOLD = 80
DILATE = 0


def extract(model, dataset, output_dir=None, layer='auto', input_key="img", mask_key="gt", evaluate=True, overlay=False):
    """

    :param model: A torch.nn.Module class
    :param dataset: A torch.utils.data.Dataset class
    :param output_dir: Save dir for the attention maps, if set to None then attention maps won't be saved
    :param layer: The layers to extract the attention maps from (auto: chooses last conv layer, 'layer name': extracts attentions from layer with this name)
    :param input_key: The dict key of a dataset batch that corresponds to the input
    :param mask_key: The dict key of a dataset batch that corresponds to the mask / ground truth / segmentation
    :param evaluate: If the dataset should be evaluated
    :param overlay: If the attention maps should be overlaid on top of the input
    :return:
    """
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # output_dir = output_dir + "/" + layer
        # Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset_len = dataset.__len__()
    model.eval()
    model_base = type(model).__bases__[0]
    model_GCAM = grad_cam.create_grad_cam(model_base)(model=model, target_layers=layer)
    model_GBP = grad_cam.create_guided_back_propagation(model_base)(model=model) # TODO: Bugged

    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = time.time()
    scores = defaultdict(list)

    with torch.enable_grad():
        for i, batch in enumerate(data_loader):
            print_progress(start, i, dataset_len, batch_size)
            image = batch[input_key]
            _ = model_GCAM.forward(image)
            _ = model_GBP.forward(image)
            is_ok = model_GCAM.model.get_ok_list()
            # print("is_ok: {}".format(is_ok))
            #cv2.imwrite("gt.png", batch[mask_key].squeeze().numpy())

            if True in is_ok: # Only if object are detected
                model_GBP.backward()
                attention_map_GBP = model_GBP.generate()
                model_GCAM.backward()
                attention_map_GCAM = model_GCAM.generate()

            # TODO: Evaluation machen
            # TODO: Guided-Backpropagation (Fehlerhaft)
            # TODO: Save images with low overlap score
            # TODO: Evaluations scores wird geraden über alle layers berechnet -> Muss für jede einzeln berechnet werden

            #print("attention_map_GCAM: {}".format(attention_map_GCAM))

            for layer_name in attention_map_GCAM.keys():
                print("layer_name: ", layer_name)
                for j in range(batch_size):
                    if is_ok[j]:
                        layer_output_dir = output_dir + "/" + layer_name
                        Path(layer_output_dir).mkdir(parents=True, exist_ok=True)
                        map_GCAM = attention_map_GCAM[layer_name][j]
                        #check_nans(attention_map_GCAM[j], i, j)
                        if overlay:
                            img = image[j].squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                        else:
                            img = None
                        if evaluate:
                            mask = batch[mask_key][j].squeeze().cpu().numpy()
                            score = evaluate_image(map_GCAM, mask)
                            scores[layer_name].append(score)
                            if output_dir is not None:
                                # save_attention_map(filename="/visinf/projects_students/shared_vqa/pythia/attention_maps/gradcam/" + str(annId) + ".npy", attention_map=map_GCAM_j)
                                save_gcam(filename=layer_output_dir + "/attention_map_" + str(i * batch_size + j) + "_score_" + str(round(score * 100)) + ".png", gcam=map_GCAM, data=img)
                                # save_guided_gcam(filename="results/guided-gcam/attention_map_" + str(i * batch_size + j) + ".png", gcam=map_GCAM_j, guided_bp=map_GBP_j)
                                print("Attention map saved.")
                    else:
                        print("Warning: No class detected in the {}-th image of batch {}!".format(j, i))
                        #save_image(batch["filepath"][j], i * batch_size + j, result_dir)
                        if evaluate:
                            scores[layer_name].append(0)

    if evaluate:
        avg_scores = {}
        for layer_name in scores.keys():
            avg_scores[layer_name] = sum(scores[layer_name]) / len(scores[layer_name])
            print("average score {}: {}%".format(layer_name, round(avg_scores[layer_name]*100, 2)))
        if output_dir is not None:
            #np.savetxt(output_dir + "/scores.txt", scores)
            #np.save(output_dir + "/scores", scores)
            with open(output_dir + '/scores.pickle', 'wb') as handle:
                pickle.dump([avg_scores, scores], handle, protocol=pickle.HIGHEST_PROTOCOL)
            # TODO: Save as pickle
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

def save_image(filepath, index, result_dir):
    raw_image = cv2.imread(filepath)
    cv2.imwrite(result_dir + "attention_map_" + str(index) + "_None.png", raw_image)

def evaluate_image(attention_map, ground_truth): # This evaluation is not suposed to be IoU
    #attention_map = attention_map.squeeze().cpu().numpy()
    #ground_truth = ground_truth.squeeze().cpu().numpy()
    ground_truth = _dilate_mask(ground_truth)
    ground_truth = _resize_image(ground_truth, attention_map)
    ground_truth = np.array(ground_truth, dtype=np.uint8)
    attention_map -= np.min(attention_map) # attention_map.min()
    attention_map /= np.max(attention_map) # attention_map.max()
    attention_map *= 255.0
    attention_map = np.array(attention_map, dtype=np.uint8)
    attention_map[attention_map < ATTENTION_THRESHOLD] = 0
    # TODO: Non-linear weighting
    overlap_percentage = cv2.bitwise_and(attention_map, attention_map, mask=ground_truth)
    attention_map = np.sum(attention_map)
    overlap_percentage = np.sum(overlap_percentage) / attention_map
    #print("overlap_percentage: {}%".format(round(overlap_percentage*100, 2)))

    return overlap_percentage

def _dilate_mask(mask):
    if DILATE > 0:
        kernel = np.ones((DILATE, DILATE), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)
    return mask

def _resize_image(mask, gcam):
    if not (np.shape(mask) == np.shape(gcam)):
        return cv2.resize(mask, np.shape(gcam))
    else:
        return mask

def print_layer_names(model, full=False, print_names=False):
    if not full:
        module_names = list(model.named_modules())[0]
    else:
        #module_names = list(model.named_modules())
        module_names = np.asarray(list(model.named_modules()))[:, 0]
    if print_names:
        print(module_names)
    with open("../module_names.txt", "w") as output:
        output.write(str(module_names))

if __name__ == "__main__":
    # from data.coco_yolo_dataset import CocoYoloDataset as Dataset
    # from models.yolo_model import YoloModel as Model
    # from data.tumor_seg_dataset import TumorDataset as Dataset
    # from models.tumor_seg_model import TumorSegModel as Model
    from models.unet_seg_example.unet_seg_dataset import UnetSegDataset as Dataset
    from models.unet_seg_example.unet_seg_model import UnetSegModel as Model

    DEVICE = "cuda" # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(device=DEVICE)
    model = Model(device=DEVICE)

    #print_layer_names(model)

    #evaluate_dataset(model, dataset, layer='model.outc.conv')
    #layers = ['model.up4', 'model.up3', 'model.up2', 'model.up1', 'model.down4', 'model.down3', 'model.down2', 'model.down1']
    # layers = ['model.up4.conv.double_conv.1',
    #           'model.up3.conv.double_conv.1',
    #           'model.up2.conv.double_conv.1',
    #           'model.up1.conv.double_conv.1',
    #           'model.down4.maxpool_conv.1.double_conv.1',
    #           'model.down3.maxpool_conv.1.double_conv.1',
    #           'model.down2.maxpool_conv.1.double_conv.1',
    #           'model.down1.maxpool_conv.1.double_conv.1']
    #
    # for layer in layers:
    #     evaluate_dataset(model, dataset, "../results", layer=layer)