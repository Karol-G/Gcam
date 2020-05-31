import torch
from torch import nn
from gcam.backends.base import _BaseWrapper


class GuidedBackPropagation(_BaseWrapper):

    def __init__(self, model, postprocessor=None, retain_graph=False):
        """
        "Striving for Simplicity: the All Convolutional Net"
        https://arxiv.org/pdf/1412.6806.pdf
        Look at Figure 1 on page 8.
        """
        super(GuidedBackPropagation, self).__init__(model, postprocessor=postprocessor, retain_graph=retain_graph)

    def _register_hooks(self):
        """Registers the backward hooks to the layers."""
        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        self.remove_hook(forward=True, backward=True)
        for module in self.model.named_modules():
            self.backward_handlers.append(module[1].register_backward_hook(backward_hook))

    def forward(self, data):
        """Calls the forward() of the base."""
        self._register_hooks()
        self.data = data.requires_grad_()
        return super(GuidedBackPropagation, self).forward(self.data)

    def generate(self):
        """Generates an attention map."""
        # try:
        attention_map = self.data.grad.clone()
        self.data.grad.zero_()
        B, _, *data_shape = attention_map.shape
        #attention_map = attention_map.view(B, self.channels, -1, *data_shape)
        attention_map = attention_map.view(B, 1, -1, *data_shape)
        attention_map = torch.mean(attention_map, dim=2)  # TODO: mean or sum?
        attention_map = attention_map.repeat(1, self.output_channels, *[1 for _ in range(self.input_dim)])
        attention_map = self._normalize_per_channel(attention_map)
        attention_map = attention_map.cpu().numpy()
        attention_maps = {}
        attention_maps[""] = attention_map
        return attention_maps
        # except RuntimeError:
        #     raise RuntimeError("Number of set channels ({}) is not a multiple of the feature map channels ({})".format(self.channels, attention_map.shape[1]))
