import torch
import numpy as np
from torch import nn
from gcam.grad_cam.backends.base import create_base_wrapper

def create_guided_back_propagation(base):
    class GuidedBackPropagation(create_base_wrapper(base)):
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
                self.handlers.append(module[1].register_backward_hook(backward_hook))

        def forward(self, data, data_shape):
            self.data = data.requires_grad_()
            return super(GuidedBackPropagation, self).forward(self.data)

        def generate(self):
            attention_map = self.data.grad.clone()
            self.data.grad.zero_()
            attention_map = attention_map.cpu().numpy().transpose(0, 2, 3, 1)
            attention_map = np.mean(attention_map, axis=3)
            attention_maps = {}
            attention_maps[""] = attention_map
            return attention_maps
    return GuidedBackPropagation
