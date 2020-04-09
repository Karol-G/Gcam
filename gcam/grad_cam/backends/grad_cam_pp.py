from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from gcam.grad_cam.backends.grad_cam import create_grad_cam

def create_grad_cam_pp(base):
    class GradCamPP(create_grad_cam(base)):
        """
        "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
        https://arxiv.org/pdf/1610.02391.pdf
        Look at Figure 2 on page 4
        """

        def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False):
            super(GradCamPP, self).__init__(model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph)

        def _select_highest_layer(self):
            # TODO: Does not always select highest layer
            # TODO: Rename method
            fmap_list, grad_list = [], []
            module_names = []
            for name, _ in self.model.named_modules():
                module_names.append(name)
            module_names.reverse()
            found_valid_layer = False

            for i in range(self.logits.shape[0]):
                counter = 0
                for layer in module_names:
                    try:
                        fmaps = self._find(self.fmap_pool, layer)
                        np.shape(fmaps)  # Throws error without this line, I have no idea why...
                        fmaps = fmaps[i]
                        grads = self._find(self.grad_pool, layer)[i]
                        nonzeros = np.count_nonzero(grads.detach().cpu().numpy())
                        self._compute_grad_weights(grads)
                        if nonzeros == 0 or not isinstance(fmaps, torch.Tensor) or not isinstance(grads, torch.Tensor):
                            counter += 1
                            continue
                        print("Dismissed the last {} module layers (Note: This number can be inflated if the model contains many nested module layers)".format(counter))
                        print("Selected module layer: {}".format(layer))
                        fmap_list.append(self._find(self.fmap_pool, layer)[i])
                        grads = self._find(self.grad_pool, layer)[i]
                        grad_list.append(grads)
                        found_valid_layer = True
                        break
                    except ValueError:
                        counter += 1
                    except RuntimeError:
                        counter += 1
                    except IndexError:
                        counter += 1

            if not found_valid_layer:
                raise ValueError("Could not find a valid layer")

            return layer, fmap_list, grad_list

        def generate(self):
            if self._target_layers == "auto":
                layer, fmaps, grads = self._select_highest_layer()
                self._check_hooks(layer)
                attention_maps = []
                for i in range(self.logits.shape[0]):
                    attention_map = self._generate_helper(fmaps[i].unsqueeze(0), grads[i].unsqueeze(0))
                    attention_map = attention_map.squeeze().cpu().numpy()
                    attention_maps.append(attention_map)
                attention_maps = {layer: attention_maps}
            else:
                attention_maps = {}
                for layer in self.target_layers:
                    self._check_hooks(layer)
                    attention_maps_tmp = self._extract_attentions(str(layer))
                    attention_maps[layer] = attention_maps_tmp
            return attention_maps

        def _extract_attentions(self, layer):
            fmaps = self._find(self.fmap_pool, layer)
            grads = self._find(self.grad_pool, layer)
            #weights = self._compute_grad_weights(grads)
            gcam_tensor = self._generate_helper(fmaps, grads)
            attention_maps = []
            for i in range(self.logits.shape[0]):
                attention_map = gcam_tensor[i].unsqueeze(0)
                attention_map = attention_map.squeeze().cpu().numpy()
                attention_maps.append(attention_map)
            return attention_maps

        def _generate_helper(self, fmaps, grads):
            b, k, u, v = grads.size()

            alpha_num = grads.pow(2)
            alpha_denom = grads.pow(2).mul(2) + \
                          fmaps.mul(grads.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

            alpha = alpha_num.div(alpha_denom + 1e-7)
            #positive_gradients = F.relu(self.logits.exp() * grads)  # TODO: self.logits only works for segmentation otherwise need to take prob of id
            positive_gradients = F.relu(10*grads)  # TODO: self.logits only works for segmentation otherwise need to take prob of id
            weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

            attention_map = (weights * fmaps).sum(1, keepdim=True)
            attention_map = F.relu(attention_map)
            #attention_map = F.upsample(attention_map, size=(224, 224), mode='bilinear', align_corners=False)
            attention_map_min, attention_map_max = attention_map.min(), attention_map.max()  # TODO: Make compatible i´with batch size bigger 1
            attention_map = (attention_map - attention_map_min).div(attention_map_max - attention_map_min).data  # TODO: Make compatible i´with batch size bigger 1
            # TODO: Remove .data and convert to numpy

            return attention_map

    return GradCamPP