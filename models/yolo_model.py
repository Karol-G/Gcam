import torch
import numpy as np
from itertools import compress
from .yolo.models import Darknet
from .yolo.utils.utils import non_max_suppression, load_classes
from torch.nn.utils.rnn import pad_sequence

MODEL_DEF = "models/yolo/config/yolov3.cfg"
WEIGHTS_PATH = "models/yolo/weights/yolov3.weights"
CLASS_PATH = "models/yolo/data/coco.names"
CONF_THRES = 0.8
NMS_THRES = 0.4
IMG_SIZE = 416
CATEGORY_NAME = 'person'

class YoloModel(torch.nn.Module):

    def __init__(self, device):
        super(YoloModel, self).__init__()
        self.device = device
        self.model = Darknet(MODEL_DEF, img_size=IMG_SIZE).to(self.device)
        self.model.load_darknet_weights(WEIGHTS_PATH)
        self.model.eval()
        self.classes = load_classes(CLASS_PATH)


    def forward(self, batch):
        detections = self.model(batch)
        detections = non_max_suppression(detections, CONF_THRES, NMS_THRES)
        self.ids, self._probs, self.probs, self.pred_classes, self.is_ok = [], [], [], [], []

        for det in detections:
            if det is None:
                self._save_results(False, [], [], [])
                continue
            ids, probs, pred_classes = self._extract_results(det)
            if not CATEGORY_NAME in pred_classes:
                self._save_results(False, [], [], [])
                continue
            self._save_results(True, ids, pred_classes, probs)

        if not self._probs:
            return torch.tensor([-1])

        self.id_pos = [x.index(CATEGORY_NAME) for x in list(compress(self.pred_classes, self.is_ok))]
        self._probs = pad_sequence(self._probs)
        self._probs = torch.transpose(self._probs, 0, 1)
        return self._probs

    def get_probs(self):
        return self.probs

    def get_ids(self):
        return self.ids

    def get_classes(self):
        return self.pred_classes

    def get_category_id_pos(self):
        return self.id_pos

    def get_ok_list(self):
        return self.is_ok

    def is_backward_ready(self):
        return False

    def _extract_results(self, det):
        det = det[:, 5:]
        det = det[det[:, 0].argsort()]
        det = torch.flip(det, [0])
        ids = det[:, 1:].squeeze(1)
        probs = det[:, :1].squeeze(1)
        pred_classes = [self.classes[x] for x in ids.long().detach().cpu().numpy()]
        return ids, probs, pred_classes

    def _save_results(self, is_ok, ids, pred_classes, probs):
        self.is_ok.append(is_ok)
        self.ids.append(ids)
        self.pred_classes.append(pred_classes)
        self.probs.append(probs)
        if is_ok:
            self._probs.append(probs)