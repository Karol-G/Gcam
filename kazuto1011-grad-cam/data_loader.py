from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import cv2
import pylab

# TODO: Mit ImageDataset Klasse in in main.py mergen?
# TODO: Checken wie man nur daten batch weise oder so läd
class TODO_GIVE_NAME():
    def __init__(self, annotation_filepath, dataset_path):
        self.annotation_filepath = annotation_filepath
        self.dataset_path = dataset_path

        # initialize COCO api for instance annotations
        self.coco=COCO(self.annotation_filepath)

    def get_image(self, img_id):
        img_infos = self.coco.loadImgs([img_id])[0]
        return io.imread(self.dataset_path+img_infos['file_name'])

    def get_ground_truth(self, img, img_id, category_name):
        ground_truth = np.zeros((img.shape[0], img.shape[1]))
        contours = self.get_contours(img_id, category_name)
        for contour in contours:
            contour = contour.astype('int32')
            cv2.fillPoly(ground_truth, [contour], 255)
        return ground_truth

    def get_contours(self, img_id, category_name):
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



img_id = 97211
annotation_filepath = '/home/karol/Documents/coco/annotations_trainval2014/annotations/instances_train2014.json'
dataset_path = '/home/karol/Documents/coco/train2014/'
loader = TODO_GIVE_NAME(annotation_filepath, dataset_path)
img = loader.get_image(97211)
ground_truth = loader.get_ground_truth(img, img_id, 'person')
plt.imshow(ground_truth)
plt.show()


# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# print("cats: {}".format(cats))
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
# get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# imgIds = coco.getImgIds(catIds=catIds );
# imgIds = coco.getImgIds(imgIds = [324158])
# img_id = 97211
# img_infos = coco.loadImgs([img_id])[0]
# print("img_infos: {}".format(img_infos))
# # load and display image
# # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# # use url to load image
# I = io.imread(dataset_path+img_infos['file_name'])
# #I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()
#
# # load and display instance annotations
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=[img_id])
# anns = coco.loadAnns(annIds)
# ax = plt.gca()
# ax.set_autoscale_on(False)
# polygons = []
# contours = []
# color = []
# for ann in anns:
#     print("ann: {}".format(ann))
#     c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
#     if 'segmentation' in ann:
#         if type(ann['segmentation']) == list:
#             # polygon
#             print("ann['segmentation']: {}".format(ann['segmentation']))
#             for seg in ann['segmentation']:
#                 poly = np.array(seg).reshape((int(len(seg)/2), 2))
#                 polygons.append(Polygon(poly))
#                 contours.append(poly)
#                 color.append(c)
# p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
# ax.add_collection(p)
# p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
# ax.add_collection(p)
# plt.show()