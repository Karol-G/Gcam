import torch
import numpy as np
import os
from .tumor_seg.bts.model import DynamicUNet

FILTER_LIST = [16,32,64,128,256]
MODEL_NAME = f"UNet-{FILTER_LIST}.pt"
STATE_DICT_PATH = '/visinf/home/vilab22/Documents/RemoteProjects/cnn_interpretability/models/tumor_seg/saved_models/'
THRESHOLD = 0.5

class TumorSegModel(torch.nn.Module):

    def __init__(self, device):
        super(TumorSegModel, self).__init__()
        self.device = device
        self.model = DynamicUNet(FILTER_LIST).to(self.device)
        self.restore_model(os.path.join(STATE_DICT_PATH, MODEL_NAME))
        self.model.eval()

    def restore_model(self, path):
        """ Loads the saved model and restores it to the "model" object.
        Loads the model based on device used for computation.(CPU/GPU)
        Follows the best method recommended by Pytorch
        Link: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
        Parameters:
            path(str): The file location where the model is saved.
        Returns:
            None
        """
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)

    def forward(self, batch):
        self.output = self.model(batch)
        self.mask = (self.output > THRESHOLD)
        self.output = self.output * self.mask
        self.output[self.output != 0] = 1

        output_numpy = self.output.detach().cpu().numpy()
        self.classes = []
        for x in output_numpy:
            nonzero = np.count_nonzero(x)
            if nonzero > 0:
                self.classes.append(["tumor"])
            else:
                self.classes.append(["no_tumor"])

        # self.classes = []
        # self.ids = []
        # for x in self.output:
        #     x = torch.max(x)
        #     if torch.sum(x) > 0:
        #         self.classes.append(["tumor"])
        #     else:
        #         self.classes.append(["no_tumor"])
        #     x = x.unsqueeze(0)
        #     self.ids.append(x)
        # self.ids = torch.stack(self.ids)
        # print("self.ids: {}".format(self.ids))
        # print("self.ids shape: {}".format(self.ids.shape))

        # _probs = []
        # for x in self.output:
        #     x = torch.flatten(x)
        #     print("x shape: {}".format(x.shape))
        #     classify = torch.nn.Linear(x.shape[0], 1).to(self.device)
        #     classify.weight.data.fill_(1.0)
        #     x = classify(x)
        #     _probs.append(x)
        #     print("x: {}".format(x))
        # _probs = torch.stack(_probs)
        # print("_probs: {}".format(_probs))

        return self.output

    def get_classes(self):
        return self.classes

    def is_backward_ready(self):
        return True

    def get_mask(self):
        return self.mask