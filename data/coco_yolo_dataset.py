import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
from torch.utils.data import Dataset
from data.cocoapi_master.PythonAPI.pycocotools.coco import COCO
from models.yolo.utils.datasets import pad_to_square, resize

IMG_SIZE = 416

class CocoYoloDataset(Dataset):
    def __init__(self, annotation_filepath, dataset_path, device):
        self.annotation_filepath = annotation_filepath
        self.dataset_path = dataset_path
        self.device = device

        # initialize COCO api for instance annotations
        self.coco=COCO(self.annotation_filepath)
        self.imgIds = self.coco.getImgIds()

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, index):
        img_id = self.imgIds[index]
        img_infos = self.coco.loadImgs([img_id])[0]
        filepath = self.dataset_path+img_infos['file_name']
        img = Image.open(filepath)
        img.save("test.png")
        if len(np.shape(img)) == 2:
            print("Convert to RGB")
            img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img, _ = pad_to_square(img, 0)
        img = resize(img, IMG_SIZE)
        img = img.to(self.device)
        return {"img": img, "gt": self._get_ground_truth(img=img, img_id=img_id, category_name="person"), "filepath": filepath}

    def _get_ground_truth(self, img, img_id, category_name=""):
        ground_truth = np.zeros((img.shape[0], img.shape[1]))
        contours = self._get_contours(img_id, category_name)
        for contour in contours:
            contour = contour.astype('int32')
            cv2.fillPoly(ground_truth, [contour], 255)
        return ground_truth

    def _get_contours(self, img_id, category_name):
        if category_name == "":
            annIds = self.coco.getAnnIds(imgIds=[img_id])
        else:
            annIds = self.coco.getAnnIds(imgIds=[img_id], catIds=[self._get_category_id(category_name)])
        anns = self.coco.loadAnns(annIds)
        contours = []
        for ann in anns:
            if 'segmentation' in ann and type(ann['segmentation']) == list:
                for seg in ann['segmentation']:
                    contour = np.array(seg).reshape((int(len(seg)/2), 2))
                    contours.append(contour)
        return np.asarray(contours)

    def _get_category_id(self, category_name):
        cats = self.coco.loadCats(self.coco.getCatIds())
        for cat in cats:
            if cat['name'] == category_name:
                return cat['id']

class CocoYoloDatasetSingle(Dataset):
    def __init__(self, annotation_filepath, dataset_path, device):
        self.device = device

    def __len__(self):
        return 1

    def __getitem__(self, index):
        img = Image.open("/visinf/home/vilab22/Documents/RemoteProjects/cnn_interpretability/models/yolo/data/samples/dog.jpg")
        img.save("test.png")
        if len(np.shape(img)) == 2:
            print("Convert to RGB")
            img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img, _ = pad_to_square(img, 0)
        img = resize(img, IMG_SIZE)
        img = img.to(self.device)
        return {"img": img, "gt": ""}