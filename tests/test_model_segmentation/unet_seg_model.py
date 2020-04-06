import torch
import os
from tests.test_model_segmentation.model.unet.unet_model import UNet

current_path = os.path.dirname(os.path.abspath(__file__))
CECKPOINT_PATH = os.path.join(current_path, 'model/CHECKPOINT.pth')
THRESHOLD = 0.5

class UnetSegModel(torch.nn.Module):

    def __init__(self, device):
        super(UnetSegModel, self).__init__()
        self.model = UNet(n_channels=3, n_classes=1)
        self.model.load_state_dict(torch.load(CECKPOINT_PATH, map_location=device))
        self.model.to(device=device)
        self.model.eval()
        self.device = device

    def forward(self, batch):
        output_batch = self.model(batch)
        probs_batch = torch.sigmoid(output_batch)
        # mask_batch = self.apply_threshold(probs_batch, THRESHOLD)
        mask_batch = probs_batch

        self.classes, self.is_ok = [], []
        for mask in mask_batch:
            if torch.sum(mask) > 0:
                self.classes.append(["car"])
                self.is_ok.append(True)
            else:
                self.classes.append(["no_car"])
                self.is_ok.append(False)

        return mask_batch

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
