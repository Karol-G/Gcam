import numpy as np
import cv2
from torch import nn
from gcam.backends.grad_cam import GradCAM
from gcam.backends.guided_backpropagation import GuidedBackPropagation


class GuidedGradCam(nn.Module):
    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False, dim=2):
        super(GuidedGradCam, self).__init__()
        self.model_GCAM = GradCAM(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim)
        self.model_GBP = GuidedBackPropagation(model=model, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim)

    def forward(self, data, data_shape):
        self.output_GCAM = self.model_GCAM.forward(data.clone(), data_shape)
        self.output_GBP = self.model_GBP.forward(data.clone(), data_shape)
        return self.output_GCAM

    def backward(self, output=None, label=None):
        self.model_GCAM.backward(output=self.output_GCAM, label=label)
        self.model_GBP.backward(output=self.output_GBP, label=label)

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
        # # del self.output_GCAM
        # # del self.output_GBP
        # # gc.collect()
        # # torch.cuda.empty_cache()
        return attention_map_GCAM
