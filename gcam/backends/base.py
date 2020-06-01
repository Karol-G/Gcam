import torch
import numpy as np
from torch.nn import functional as F
from gcam import gcam_utils


class _BaseWrapper():

    def __init__(self, model, postprocessor=None, retain_graph=False):
        """A base wrapper of common functions for the backends."""
        self.device = next(model.parameters()).device
        self.retain_graph = retain_graph
        self.model = model
        self.forward_handlers = []
        self.backward_handlers = []
        self.postprocessor = postprocessor

    def generate_attention_map(self, batch, label):
        """Handles the generation of the attention map from start to finish."""
        output = self.forward(batch)
        self.backward(label=label)
        attention_map = self.generate()
        return output, attention_map, self.output_batch_size, self.output_channels, self.output_shape

    def forward(self, batch):
        """Calls the forward() of the model."""
        self.model.zero_grad()
        self.logits = self.model.model_forward(batch)
        self._extract_metadata(batch, self.logits)
        self._set_postprocessor_and_label(self.logits)
        self.remove_hook(forward=True, backward=False)
        return self.logits

    def backward(self, label=None):
        """Applies postprocessing and class discrimination on the model output and then backwards it."""
        if label is None:
            label = self.model.gcam_dict['label']

        processed_logits = self.post_processing(self.postprocessor, self.logits)
        self.mask = self._mask_output(processed_logits, label)

        if self.mask is None:
            self.logits.backward(gradient=self.logits, retain_graph=self.retain_graph)
        else:
            self.logits.backward(gradient=self.mask, retain_graph=self.retain_graph)
        self.remove_hook(forward=True, backward=True)

    def post_processing(self, postprocessor, output):
        """The postprocessor is applied on the model output from calling forward which is then passed to the class discriminator. It converts the raw logit output from the model to a usable form for the class discriminator. The postprocessor is only applied internally, the final output given to the user is not effected.

                    None: No postprocessing is applied.

                    'sigmoid': Applies the sigmoid function.

                    'softmax': Applies softmax function.

                    (A function): Applies a given function to the output.

        """
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
        """Creates a binary mask that is later applied to the postprocessed output."""
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

    def _extract_metadata(self, input, output):  # TODO: Does not work for classification output (shape: (1, 1000)), merge with the one in gcam_inject
        """Extracts metadata like batch size, number of channels and the data shape from the output batch."""
        self.input_dim = len(input.shape[2:])
        self.output_batch_size = output.shape[0]
        if self.model.gcam_dict['channels'] == 'default':
            self.output_channels = output.shape[1]
        else:
            self.output_channels = self.model.gcam_dict['channels']
        if self.model.gcam_dict['data_shape'] == 'default':
            self.output_shape = output.shape[2:]
        else:
            self.output_shape = self.model.gcam_dict['data_shape']

    def _normalize_per_channel(self, attention_map):
        if torch.min(attention_map) == torch.max(attention_map):
            return torch.zeros(attention_map.shape)
        # Normalization per channel
        B, C, *data_shape = attention_map.shape
        attention_map = attention_map.view(B, C, -1)
        attention_map_min = torch.min(attention_map, dim=2, keepdim=True)[0]
        attention_map_max = torch.max(attention_map, dim=2, keepdim=True)[0]
        attention_map -= attention_map_min
        attention_map /= (attention_map_max - attention_map_min)
        attention_map = attention_map.view(B, C, *data_shape)
        return attention_map

    def generate(self):
        """Generates an attention map."""
        raise NotImplementedError

    def remove_hook(self, forward, backward):
        """
        Remove all the forward/backward hook functions
        """
        if forward:
            for handle in self.forward_handlers:
                handle.remove()
            self.forward_handlers = []
        if backward:
            for handle in self.backward_handlers:
                handle.remove()
            self.backward_handlers = []

    def layers(self, reverse=False):
        """Returns the layers of the model. Optionally reverses the order of the layers."""
        return gcam_utils.get_layers(self.model, reverse)

    def _set_postprocessor_and_label(self, output):
        if self.postprocessor is None:
            if output.shape[0] == self.output_batch_size and len(output.shape) == 2:  # classification
                self.postprocessor = "softmax"
            elif output.shape[0] == self.output_batch_size and len(output.shape) == 4 and output.shape[1] == 1:  # 2D segmentation
                self.postprocessor = "sigmoid"
            elif output.shape[0] == self.output_batch_size and len(output.shape) == 4 and output.shape[1] > 1:  # 3D segmentation (nnUNet)
                self.postprocessor = torch.nn.Softmax(dim=2)
        if self.model.gcam_dict['label'] is None:
            if output.shape[0] == self.output_batch_size and len(output.shape) == 2:  # classification
                self.model.gcam_dict['label'] = "best"
