#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp
import os, os.path

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import skimage.io as io
from pycocotools.coco import COCO

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

#os.environ['CUDA_VISIBLE_DEVICES']='2'

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

class ImageDataset():
    def __init__(self, annotation_filepath, dataset_path):
        self.annotation_filepath = annotation_filepath
        self.dataset_path = dataset_path

        # initialize COCO api for instance annotations
        self.coco=COCO(self.annotation_filepath)
        self.imgIds = self.coco.getImgIds()

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, index):
        imgId = self.imgIds[index]
        img = np.asarray(self.get_image(imgId))
        img_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(img)
        return {"img": img_tensor, "gt": self.get_ground_truth(img, imgId)}

    def get_image(self, img_id):
        img_infos = self.coco.loadImgs([img_id])[0]
        return io.imread(self.dataset_path+img_infos['file_name'])

    def get_ground_truth(self, img, img_id, category_name=""):
        ground_truth = np.zeros((img.shape[0], img.shape[1]))
        contours = self.get_contours(img_id, category_name)
        for contour in contours:
            contour = contour.astype('int32')
            cv2.fillPoly(ground_truth, [contour], 255)
        return ground_truth

    def get_contours(self, img_id, category_name):
        if category_name == "":
            annIds = self.coco.getAnnIds(imgIds=[img_id])
        else:
            annIds = self.coco.getAnnIds(imgIds=[img_id], catIds=[self.get_category_id(category_name)])
        anns = self.coco.loadAnns(annIds)
        contours = []
        for ann in anns:
            if 'segmentation' in ann and type(ann['segmentation']) == list:
                for seg in ann['segmentation']:
                    contour = np.array(seg).reshape((int(len(seg)/2), 2))
                    contours.append(contour)
        return np.asarray(contours)

    def get_category_id(self, category_name):
        cats = self.coco.loadCats(self.coco.getCatIds())
        for cat in cats:
            if cat['name'] == category_name:
                return cat['id']

# class ImageDataset(Dataset):
#     def __init__(self, path):
#         annotation_filepath = '/data/vilab22/vqa2/coco-annotations/instances_train2014.json'
#         dataset_path = '/data/vilab22/vqa2/train2014/'
#         loader = ImageLoader(annotation_filepath, dataset_path)
#         # self.images = []
#         # self.raw_images = []
#         #
#         # valid_images = [".jpg", ".png", ".jpeg"]
#         # for f in os.listdir(path):
#         #     ext = os.path.splitext(f)[1]
#         #     if ext.lower() not in valid_images:
#         #         continue
#         #     image, raw_image = preprocess(os.path.join(path, f))
#         #     self.images.append(image)
#         #     self.raw_images.append(raw_image)
#
#     def __len__(self):
#         # return len(self.images.shape)
#
#     def __getitem__(self, index):
#         # return self.images[index]
#
#     def get_images(self):
#         # return self.images
#
#     def get_raw_images(self):
#         # return self.raw_images


def main(image_paths, target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """

    if not cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    #print(*list(model.named_modules()), sep='\n')
    model.to(device)
    model.eval()

    annotation_filepath = '/visinf/projects_students/shared_vqa/mscoco/coco-annotations/instances_train2014.json'
    dataset_path = '/visinf/projects_students/shared_vqa/mscoco/train2014/'
    dataset = ImageDataset(annotation_filepath, dataset_path)
    #dataset = ImageDataset(image_paths[0])

    # Images
    #images = dataset.get_images()
    images = [dataset.__getitem__(0)["img"]]
    #raw_images = dataset.get_raw_images()
    # print("Images:")
    # for i, image_path in enumerate(image_paths):
    #     print("\t#{}: {}".format(i, image_path))
    #     image, raw_image = preprocess(image_path)
    #     images.append(image)
    #     raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)

    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layers=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # # Guided Backpropagation
            # save_gradient(filename=osp.join(output_dir, "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),),gradient=gradients[j],)

            # # Grad-CAM
            # save_gradcam(filename=osp.join(output_dir, "{}-{}-gradcam-{}-{}.png".format(j, arch, target_layer, classes[ids[j, i]]),), gcam=regions[j, 0], raw_image=raw_images[j],)

            # Guided Grad-CAM
            save_gradient(filename=osp.join(output_dir, "{}-{}-guided_gradcam-{}-{}.png".format(j, arch, target_layer, classes[ids[j, i]]),), gradient=torch.mul(regions, gradients)[j],)

    print("Finished execution.")

def evaluate(gradients, ground_truth):
    gradients = gradients.cpu().numpy().transpose(1, 2, 0)
    gradients -= gradients.min()
    gradients /= gradients.max()
    gradients *= 255.0
    gradients = np.array(gradients, dtype=np.uint8)
    # TODO: Erode or dilate ground truth
    # TODO: Resize ground truth to image size of gradients
    ground_truth_inv = 255 - ground_truth
    gradients_true = cv2.bitwise_and(gradients, gradients, mask=ground_truth)
    gradients_false = cv2.bitwise_and(gradients, gradients, mask=ground_truth_inv)
    gradients = np.sum(gradients)
    gradients_true = np.sum(gradients_true) / gradients
    gradients_false = np.sum(gradients_false) / gradients
    print("gradients_true: " + str(gradients_true) + "%")
    print("gradients_false: " + str(gradients_false) + "%")

    return gradients_true, gradients_false

if __name__ == "__main__":
    #main(["samples/cat_dog.png"], "features.35", "vgg19", 3, "./results", True)
    main(["samples/"], "features.35", "vgg19", 3, "results", True)
