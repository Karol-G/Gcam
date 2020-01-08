import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
from torch.utils.data import Dataset
from data.cocoapi_master.PythonAPI.pycocotools.coco import COCO
from models.yolo.utils.datasets import pad_to_square, resize
from models.yolo.utils.utils import load_classes

ANNOTATION_FILEPATH = '/visinf/projects_students/shared_vqa/mscoco/coco-annotations/instances_train2014.json'
DATASET_PATH = '/visinf/projects_students/shared_vqa/mscoco/train2014/'
CLASS_PATH = "models/yolo/data/coco.names"
IMG_SIZE = 416
CATEGORY_NAME = 'person'

class CocoYoloDataset(Dataset):
    def __init__(self, device):
        self.annotation_filepath = ANNOTATION_FILEPATH
        self.dataset_path = DATASET_PATH
        self.device = device

        # initialize COCO api for instance annotations
        self.coco=COCO(self.annotation_filepath)
        #coco_cat_names = self._get_all_categories()
        #yolo_cat_names = load_classes(CLASS_PATH)
        #self.categories = self._intersection(coco_cat_names, yolo_cat_names)
        self.imgIds = self.coco.getImgIds(catIds=[self._get_category_id(CATEGORY_NAME)])

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, index):
        if index == 10035: # The corresponding image is corrupted
            index = 10034
        img_id = self.imgIds[index]
        img_infos = self.coco.loadImgs([img_id])[0]
        filepath = self.dataset_path+img_infos['file_name']
        img = Image.open(filepath)
        if len(np.shape(img)) == 2:
            img = img.convert('RGB')
        gt = self._get_ground_truth(img=img, img_id=img_id, category_name=CATEGORY_NAME)
        img = transforms.ToTensor()(img)
        img, _ = pad_to_square(img, 0)
        img = resize(img, IMG_SIZE)
        img = img.to(self.device)

        return {"img": img, "gt": gt, "filepath": filepath}

    def _get_ground_truth(self, img, img_id, category_name=""):
        ground_truth = np.zeros((np.shape(img)[0], np.shape(img)[1]))
        contours = self._get_contours(img_id, category_name)
        for contour in contours:
            contour = contour.astype('int32')
            cv2.fillPoly(ground_truth, [contour], 255)
        ground_truth = cv2.resize(ground_truth, (IMG_SIZE, IMG_SIZE))
        ground_truth = np.array(ground_truth, dtype=np.uint8)
        ground_truth[ground_truth > 0] = 1
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

    # def _get_all_categories(self):
    #     cats = self.coco.loadCats(self.coco.getCatIds())
    #     coco_cat_names = []
    #     for cat in cats:
    #         coco_cat_names.append(cat['name'])
    #     return coco_cat_names

    # def _intersection(self, lst1, lst2):
    #     return list(set(lst1).intersection(lst2))

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