import torch
import numpy as np
from .yolo.models import Darknet
from .yolo.utils.utils import non_max_suppression, load_classes
from torch.nn.utils.rnn import pad_sequence

MODEL_DEF = "models/yolo/config/yolov3.cfg"
WEIGHTS_PATH = "models/yolo/weights/yolov3.weights"
CLASS_PATH = "models/yolo/data/coco.names"
CONF_THRES = 0.8
NMS_THRES = 0.4
IMG_SIZE = 416

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
        self.ids, self._probs, self.probs, self.pred_classes = [], [], [], []
        for det in detections:
            if det is None:
                self.pred_classes.append([])
                self.ids.append([])
                self.probs.append([])
                continue
            det = det[:, 5:]
            det = det[det[:,0].argsort()]
            det = torch.flip(det, [0])
            ids = det[:, 1:].squeeze(1)
            probs = det[:, :1].squeeze(1)
            self.pred_classes.append([self.classes[x] for x in ids.long().detach().cpu().numpy()])
            self.ids.append(ids)
            self._probs.append(probs)
            self.probs.append(probs)

        if not self._probs:
            return torch.tensor([-1])
        self._probs = pad_sequence(self._probs)
        self._probs = torch.transpose(self._probs, 0, 1)
        return self._probs

    def get_probs(self):
        return self.probs

    def get_ids(self):
        return self.ids

    def get_classes(self):
        return self.pred_classes

    def is_backward_ready(self):
        return False