import numpy as np
import cv2
import gc
import torch
from gcam.grad_cam.backends.grad_cam import create_grad_cam
from gcam.grad_cam.backends.guided_backpropagation import create_guided_back_propagation

def create_guided_grad_cam(base):
    class GuidedGradCam(base):
        def __init__(self, model, target_layers=None, is_backward_ready=None, postprocessor=None, retain_graph=False):
            self.model_GCAM = create_grad_cam(base)(model=model, target_layers=target_layers, is_backward_ready=is_backward_ready, postprocessor=postprocessor, retain_graph=retain_graph)
            self.model_GBP = create_guided_back_propagation(base)(model=model, is_backward_ready=is_backward_ready, postprocessor=postprocessor, retain_graph=retain_graph)

        def forward(self, data, data_shape):
            self.output_GCAM = self.model_GCAM.forward(data.clone(), data_shape)
            self.output_GBP = self.model_GBP.forward(data.clone(), data_shape)
            return self.output_GCAM

        def backward(self, ids=None, output=None):
            self.model_GCAM.backward(ids=ids, output=self.output_GCAM)
            self.model_GBP.backward(ids=ids, output=self.output_GBP)

        def generate(self):
            attention_map_GCAM = self.model_GCAM.generate()
            attention_map_GBP = self.model_GBP.generate()[""]
            for layer_name in attention_map_GCAM.keys():
                for i in range(len(attention_map_GCAM[layer_name])):
                    if attention_map_GBP[i].shape == attention_map_GCAM[layer_name][i].shape:
                        attention_map_GCAM[layer_name][i] = np.multiply(attention_map_GCAM[layer_name][i], attention_map_GBP[i])
                    else:
                        attention_map_GCAM_tmp = cv2.resize(attention_map_GCAM[layer_name][i], tuple(np.flip(attention_map_GBP[i].shape)))
                        attention_map_GCAM[layer_name][i] = np.multiply(attention_map_GCAM_tmp, attention_map_GBP[i])
            # # del self.output_GCAM
            # # del self.output_GBP
            # # gc.collect()
            # # torch.cuda.empty_cache()
            return attention_map_GCAM
    return GuidedGradCam
