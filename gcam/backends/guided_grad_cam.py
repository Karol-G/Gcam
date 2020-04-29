import numpy as np
from gcam.backends.grad_cam import GradCAM
from gcam.backends.guided_backpropagation import GuidedBackPropagation
from gcam import gcam_utils


class GuidedGradCam():
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False):
        self.model_GCAM = GradCAM(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph)
        self.model_GBP = GuidedBackPropagation(model=model, postprocessor=postprocessor, retain_graph=retain_graph)

    def forward(self, data):
        """Calls the forward() of the backends gbp and gcam."""
        self.output_GCAM = self.model_GCAM.forward(data.clone())
        self.output_GBP = self.model_GBP.forward(data.clone())
        return self.output_GCAM

    def backward(self, label=None):
        """Calls the backward() of the backends gbp and gcam."""
        self.model_GCAM.backward(label=label)
        self.model_GBP.backward(label=label)

    def get_registered_hooks(self):
        """Returns every hook that was able to register to a layer."""
        return self.model_GCAM.get_registered_hooks()

    def generate(self):
        """Generates an attention map."""
        attention_map_GCAM = self.model_GCAM.generate()
        attention_map_GBP = self.model_GBP.generate()[""]
        for layer_name in attention_map_GCAM.keys():
            if attention_map_GBP.shape == attention_map_GCAM[layer_name].shape:
                attention_map_GCAM[layer_name] = np.multiply(attention_map_GCAM[layer_name], attention_map_GBP)
            else:
                attention_map_GCAM_tmp = gcam_utils.interpolate(attention_map_GCAM[layer_name], attention_map_GBP.shape[2:])
                attention_map_GCAM[layer_name] = np.multiply(attention_map_GCAM_tmp, attention_map_GBP)
        return attention_map_GCAM
