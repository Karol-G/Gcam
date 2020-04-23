import numpy as np
import cv2
from gcam.backends.grad_cam import GradCAM
from gcam.backends.guided_backpropagation import GuidedBackPropagation
from gcam import gcam_utils


class GuidedGradCam():
    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False, dim=2, registered_only=False):
        self.model_GCAM = GradCAM(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim, registered_only=registered_only)
        self.model_GBP = GuidedBackPropagation(model=model, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim)

    def forward(self, data, data_shape):
        self.output_GCAM = self.model_GCAM.forward(data.clone(), data_shape)
        self.output_GBP = self.model_GBP.forward(data.clone(), data_shape)
        return self.output_GCAM

    def backward(self, label=None):
        self.model_GCAM.backward(label=label)
        self.model_GBP.backward(label=label)

    def generate(self):
        attention_map_GCAM = self.model_GCAM.generate()
        attention_map_GBP = self.model_GBP.generate()[""]
        for layer_name in attention_map_GCAM.keys():
            for i in range(len(attention_map_GCAM[layer_name])):
                if attention_map_GBP[i].shape == attention_map_GCAM[layer_name][i].shape:
                    attention_map_GCAM[layer_name][i] = np.multiply(attention_map_GCAM[layer_name][i], attention_map_GBP[i])
                else:
                    attention_map_GCAM_tmp = cv2.resize(attention_map_GCAM[layer_name][i], tuple(np.flip(attention_map_GBP[i].shape)))  # TODO: Not compatible with 3D
                    attention_map_GCAM[layer_name][i] = np.multiply(attention_map_GCAM_tmp, attention_map_GBP[i])
        return attention_map_GCAM
