import torch
from torch.nn import functional as F
from gcam.backends.grad_cam import GradCAM
from gcam import gcam_utils
from gcam.gcam_utils import prod


class GradCamPP(GradCAM):

    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False):
        """
        "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
        https://arxiv.org/abs/1710.11063
        """
        super(GradCamPP, self).__init__(model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph)

    def _generate_helper(self, fmaps, grads):
        B, C, *data_shape = grads.size()

        alpha_num = grads.pow(2)
        tmp = fmaps.mul(grads.pow(3))
        tmp = tmp.view(B, C, prod(data_shape))
        tmp = tmp.sum(-1, keepdim=True)
        if self.input_dim == 2:
            tmp = tmp.view(B, C, 1, 1)
        else:
            tmp = tmp.view(B, C, 1, 1, 1)
        alpha_denom = grads.pow(2).mul(2) + tmp
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-7)

        mask = self.mask.squeeze()
        if self.mask is None:
            prob_weights = 1
        elif len(mask.shape) == 1:
            prob_weights = self.logits.squeeze()[torch.argmax(mask)]
        else:
            prob_weights = gcam_utils.interpolate(self.logits, grads.shape[2:])  # TODO: Still removes channels...

        positive_gradients = F.relu(torch.mul(prob_weights.exp(), grads))
        weights = (alpha * positive_gradients).view(B, C, -1).sum(-1)
        if self.input_dim == 2:
            weights = weights.view(B, C, 1, 1)
        else:
            weights = weights.view(B, C, 1, 1, 1)

        attention_map = (weights * fmaps)
        attention_map = attention_map.view(B, self.output_channels, -1, *data_shape)
        attention_map = torch.sum(attention_map, dim=2)
        attention_map = F.relu(attention_map).detach()
        attention_map = self._normalize_per_channel(attention_map)

        return attention_map