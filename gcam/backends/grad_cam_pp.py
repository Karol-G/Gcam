import numpy as np
import torch
from torch.nn import functional as F
from gcam.backends.grad_cam import GradCAM


class GradCamPP(GradCAM):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False, dim=2):
        super(GradCamPP, self).__init__(model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph)
        self.dim = dim

    def _generate_helper(self, fmaps, grads):
        b, k, u, v = grads.size()

        alpha_num = grads.pow(2)
        alpha_denom = grads.pow(2).mul(2) + \
                      fmaps.mul(grads.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)

        logits = F.interpolate(self.logits, grads.shape[2:], mode="bilinear", align_corners=False)  # TODO: Only works with 2D

        positive_gradients = F.relu(torch.mul(logits.exp(), grads))  # TODO: self.logits only works for segmentation otherwise need to take prob of id
        #positive_gradients = F.relu(10 * grads)  # TODO: self.logits only works for segmentation otherwise need to take prob of id
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        attention_map = (weights * fmaps).sum(1, keepdim=True)
        attention_map = F.relu(attention_map)
        # attention_map = F.upsample(attention_map, size=(224, 224), mode='bilinear', align_corners=False)
        attention_map_min, attention_map_max = attention_map.min(), attention_map.max()  # TODO: Make compatible i´with batch size bigger 1
        attention_map = (attention_map - attention_map_min).div(attention_map_max - attention_map_min).data  # TODO: Make compatible i´with batch size bigger 1
        # TODO: Remove .data and convert to numpy

        return attention_map