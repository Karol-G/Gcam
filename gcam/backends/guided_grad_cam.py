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

    def generate_attention_map(self, batch, label):
        """Handles the generation of the attention map from start to finish."""
        output, self.output_GCAM, output_batch_size, output_channels, output_shape = self.model_GCAM.generate_attention_map(batch.clone(), label)
        _, self.output_GBP, _, _, _ = self.model_GBP.generate_attention_map(batch.clone(), label)
        attention_map = self.generate()
        return output, attention_map, output_batch_size, output_channels, output_shape

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
