import torch
import numpy as np
from torch.nn import functional as F
from gcam import gcam_utils



class _BaseWrapper():

    def __init__(self, model, postprocessor=None, retain_graph=False):
        self.device = next(model.parameters()).device
        self.retain_graph = retain_graph
        self.model = model
        self.handlers = []
        self.postprocessor = postprocessor

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, data):
        self.model.zero_grad()
        self.logits = self.model.model_forward(data)
        self._extract_metadata(self.logits)
        return self.logits

    def backward(self, label=None):
        if label is None:
            label = self.model.gcam_dict['label']

        processed_logits = self.post_processing(self.postprocessor, self.logits)
        self.mask = self._mask_output(processed_logits, label)

        if self.mask is None:
            self.logits.backward(gradient=self.logits, retain_graph=self.retain_graph)
        else:
            self.logits.backward(gradient=self.mask, retain_graph=self.retain_graph)

    def post_processing(self, postprocessor, output):
        if postprocessor is None:
            return output
        elif postprocessor == "sigmoid":
            output = torch.sigmoid(output)
        elif postprocessor == "softmax":
            output = F.softmax(output, dim=1)
        elif callable(postprocessor):
            output = postprocessor(output)
        else:
            raise ValueError("Postprocessor must be either None, 'sigmoid', 'softmax' or a postprocessor function")
        return output

    def _mask_output(self, output, label):
        if label is None:
            return None
        elif label == "best":  # Only for classification
            indices = torch.argmax(output).detach().cpu().numpy()
            mask = np.zeros(output.shape)
            np.put(mask, indices, 1)
        elif isinstance(label, int):  # Only for classification
            indices = (output == label).nonzero()
            indices = [index[0] * output.shape[1] + index[1] for index in indices]
            mask = np.zeros(output.shape)
            np.put(mask, indices, 1)
        elif callable(label):  # Can be used for everything, but is best for segmentation 2D/3D
            mask = label(output).detach().cpu().numpy()
        else:
            raise ValueError("Label must be either None, 'best', a class label index or a discriminator function")
        mask = torch.FloatTensor(mask).to(self.device)
        return mask

    def _extract_metadata(self, data):  # TODO: Does not work for classification output (shape: (1, 1000)), merge with the one in gcam_inject
        self.dim = len(data.shape[2:])
        self.batch_size = data.shape[0]
        if self.model.gcam_dict['channels'] == 'default':
            self.channels = data.shape[1]
        else:
            self.channels = self.model.gcam_dict['channels']
        if self.model.gcam_dict['data_shape'] == 'default':
            self.data_shape = data.shape[2:]
        else:
            self.data_shape = self.model.gcam_dict['data_shape']

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
