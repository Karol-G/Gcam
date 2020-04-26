import numpy as np
import torch
from torch.nn import functional as F
from gcam.backends.grad_cam import GradCAM
from gcam import gcam_utils
from gcam.gcam_utils import prod


class GradCamPP(GradCAM):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False):
        super(GradCamPP, self).__init__(model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph)

    # def _generate_helper(self, fmaps, grads):  # TODO: Is b, k, u, v 3d compatible? channel dim reducen wie bei grad cam
    #     b, k, u, v = grads.size()
    #
    #     alpha_num = grads.pow(2)
    #     tmp1 = fmaps.mul(grads.pow(3))
    #     tmp2 = tmp1.view(b, k, u * v)
    #     tmp3 = tmp2.sum(-1, keepdim=True)
    #     tmp4 = tmp3.view(b, k, 1, 1)
    #     alpha_denom = grads.pow(2).mul(2) + tmp4
    #     alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    #     alpha = alpha_num.div(alpha_denom + 1e-7)
    #
    #     mask = self.mask.squeeze()
    #     if self.mask is None:
    #         prob_weights = 1
    #     elif len(mask.shape) == 1:
    #         prob_weights = self.logits.squeeze()[torch.argmax(mask)]
    #     else:
    #         prob_weights = gcam_utils.interpolate(self.logits, grads.shape[2:])  # TODO: Removes channels
    #
    #     positive_gradients = F.relu(torch.mul(prob_weights.exp(), grads))
    #     weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)  # TODO
    #
    #     attention_map = (weights * fmaps).sum(1, keepdim=True)
    #     attention_map = F.relu(attention_map).detach()
    #     attention_map = self._normalize(attention_map)
    #
    #     return attention_map

    def _generate_helper(self, fmaps, grads):  # TODO: Bugged
        B, _, *data_shape = grads.size()

        alpha_num = grads.pow(2)
        tmp = fmaps.mul(grads.pow(3))
        tmp = tmp.view(B, self.channels, -1, *data_shape)
        tmp = torch.sum(tmp, dim=2)
        tmp = tmp.view(B, self.channels, -1)
        tmp = tmp.sum(-1, keepdim=True)
        if len(data_shape):
            tmp = tmp.view(B, self.channels, 1, 1)
        else:
            tmp = tmp.view(B, self.channels, 1, 1, 1)
        alpha_denom = grads.pow(2).mul(2) + tmp
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-7)

        mask = self.mask.squeeze()
        if self.mask is None:
            prob_weights = 1
        elif len(mask.shape) == 1:
            prob_weights = self.logits.squeeze()[torch.argmax(mask)]
        else:
            prob_weights = gcam_utils.interpolate(self.logits, grads.shape[2:])  # TODO: Removes channels

        positive_gradients = F.relu(torch.mul(prob_weights.exp(), grads))
        weights = (alpha * positive_gradients).view(B, self.channels, -1).sum(-1)
        if len(data_shape):
            weights = weights.view(B, self.channels, 1, 1)
        else:
            weights = weights.view(B, self.channels, 1, 1, 1)

        attention_map = (weights * fmaps)
        attention_map = attention_map.view(B, self.channels, -1, *data_shape)
        attention_map = torch.sum(attention_map, dim=2)
        attention_map = F.relu(attention_map).detach()
        attention_map = self._normalize(attention_map)

        return attention_map