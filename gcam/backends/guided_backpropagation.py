import torch
import numpy as np
from torch import nn
from gcam.backends.base import _BaseWrapper


class GuidedBackPropagation(_BaseWrapper):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model, postprocessor=None, retain_graph=False):
        super(GuidedBackPropagation, self).__init__(model, postprocessor=postprocessor, retain_graph=retain_graph)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            self.backward_handlers.append(module[1].register_backward_hook(backward_hook))

    def forward(self, data):
        self.data = data.requires_grad_()
        return super(GuidedBackPropagation, self).forward(self.data)

    def generate(self):
        try:
            attention_map = self.data.grad.clone()
            self.data.grad.zero_()
            B, _, *data_shape = attention_map.shape
            attention_map = attention_map.view(B, self.channels, -1, *data_shape)
            attention_map = torch.mean(attention_map, dim=2)  # TODO: mean or sum?
            attention_map = attention_map.cpu().numpy()
            attention_maps = {}
            attention_maps[""] = attention_map
            return attention_maps
        except RuntimeError:
            raise RuntimeError("Number of set channels ({}) is not a multiple of the feature map channels ({})".format(self.channels, attention_map.shape[1]))
