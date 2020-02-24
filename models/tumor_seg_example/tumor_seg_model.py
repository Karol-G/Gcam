import torch
import numpy as np
import os

from models.tumor_seg_example.model.bts.model import DynamicUNet

FILTER_LIST = [16,32,64,128,256]
MODEL_NAME = f"UNet-{FILTER_LIST}.pt"
current_path = os.path.dirname(os.path.abspath(__file__))
STATE_DICT_PATH = os.path.join(current_path, 'model/saved_models/')
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
        # self.mask = (self.output > THRESHOLD)
        # self.output = self.output * self.mask
        # self.output[self.output != 0] = 1
        self.output = self.apply_threshold(self.output, THRESHOLD)

        output_numpy = self.output.detach().cpu().numpy()
        self.classes, self.is_ok = [], []
        for x in output_numpy:
            nonzero = np.count_nonzero(x)
            if nonzero > 0:
                self.classes.append(["tumor"])
                self.is_ok.append(True)
            else:
                self.classes.append(["no_tumor"])
                self.is_ok.append(False)

        return self.output

    def get_classes(self):
        return self.classes

    def is_backward_ready(self):
        return True

    # def get_mask(self):
    #     return self.mask

    def get_ok_list(self):
        return self.is_ok

    def apply_threshold(self, output, threshold):
        mask = (output > threshold)
        output = output * mask
        output[output != 0] = 1
        return output