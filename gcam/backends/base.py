import torch
import numpy as np
from torch.nn import functional as F
from gcam import gcam_utils
from torch import nn



class _BaseWrapper(nn.Module):
    """
    Please modify forward() and backward() according to your task.
    """

    def __init__(self, model, postprocessor=None, retain_graph=False):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.retain_graph = retain_graph
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.postprocessor = postprocessor

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, data):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(data)
        return self.logits

    def backward(self, output=None, label=None):
        if output is not None:
            self.logits = output

        self.logits = self.post_processing(self.postprocessor, self.logits)
        mask = self._mask_output(output, label)

        if mask is None:
            self.logits.backward(gradient=self.logits, retain_graph=self.retain_graph)
        else:
            self.logits.backward(gradient=mask, retain_graph=self.retain_graph)

    def post_processing(self, postprocessor, output):
        if postprocessor is None:
            return output
        elif postprocessor == "sigmoid":
            output = torch.sigmoid(output)
        elif postprocessor == "softmax":
            output = F.softmax(output, dim=1)
        else:
            output = postprocessor(output)
        return output

    def _mask_output(self, output, label):
        if label is None:
            return None
        elif label == "best":
            indices = torch.argmax(output).detach().cpu().numpy()
        elif isinstance(label, int):
            indices = (output == label).nonzero()
            indices = [index[0] * output.shape[1] + index[1] for index in indices]  # TODO: Not compatible with 3D data
        else:
            indices = label(output)
        mask = np.zeros(output.shape)
        np.put(mask, indices, 1)
        mask = torch.FloatTensor(mask).to(self.device)
        return mask

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def layers(self, reverse=False):
        return gcam_utils.get_layers(self.model, reverse)
