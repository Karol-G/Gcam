from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from gcam.backends.base import _BaseWrapper

def detach_output(output):  # TODO: Is this needed? Is this correct?
    if not isinstance(output, torch.Tensor):
        tuple_list = []
        if hasattr(output, '__iter__'):
            for item in output:
                if isinstance(output, torch.Tensor):
                    tuple_list.append(item.detach())
                else:
                    tuple_list.append(detach_output(item))
            return tuple_list
        else:
            return output
    return output.detach()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, target_layers=None, postprocessor=None, retain_graph=False, registered_only=False):
        super(GradCAM, self).__init__(model, postprocessor=postprocessor, retain_graph=retain_graph)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.module_names = {}
        self._target_layers = target_layers
        if target_layers == 'full' or target_layers == 'auto':
            target_layers = np.asarray(list(self.model.named_modules()))[:, 0]
        elif isinstance(target_layers, str):
            target_layers = [target_layers]
        self.target_layers = target_layers
        self.registered_hooks = {}
        self.registered_only = registered_only

        def forward_hook(key):
            def forward_hook_(module, input, output):
                self.registered_hooks[key][0] = True
                # Save featuremaps
                self.fmap_pool[key] = detach_output(output)
                if not isinstance(output, torch.Tensor):
                    print("Cannot hook layer {} because its gradients are not in tensor format".format(key))

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                #print("grad_out: ", grad_out[0].shape)
                self.registered_hooks[key][1] = True
                # Save the gradients correspond to the featuremaps
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in model.named_modules():
            if self.target_layers is None or name in self.target_layers:
                # print("Trying to hook layer: {}".format(name))
                self.registered_hooks[name] = [False, False]
                self.module_names[module] = name
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        if len(self.data_shape) == 2:
            return F.adaptive_avg_pool2d(grads, 1)
        else:
            return F.adaptive_avg_pool3d(grads, 1)

    def forward(self, data):
        return super(GradCAM, self).forward(data)

    def generate(self):
        if self._target_layers == "auto":
            layer, fmaps, grads = self._auto_layer_selection()
            self._check_hooks(layer)
            attention_map = self._generate_helper(fmaps, grads).cpu().numpy()
            attention_maps = {layer: attention_map}
        else:
            attention_maps = {}
            for layer in self.target_layers:
                if not self.registered_only:
                    self._check_hooks(layer)
                if self.registered_hooks[layer][0] and self.registered_hooks[layer][1]:
                    attention_maps[layer] = self._extract_attentions(str(layer)).cpu().numpy()
        return attention_maps

    def _auto_layer_selection(self):
        # It's ugly but it works ;)
        module_names = self.layers(reverse=True)
        found_valid_layer = False

        counter = 0
        for layer in module_names:
            try:
                fmaps = self._find(self.fmap_pool, layer)
                grads = self._find(self.grad_pool, layer)
                nonzeros = np.count_nonzero(grads.detach().cpu().numpy())
                self._compute_grad_weights(grads)
                if nonzeros == 0 or not isinstance(fmaps, torch.Tensor) or not isinstance(grads, torch.Tensor):
                    counter += 1
                    continue
                print("Selected module layer: {}".format(layer))
                found_valid_layer = True
                break
            except ValueError:
                counter += 1
            except RuntimeError:
                counter += 1
            except IndexError:
                counter += 1

        if not found_valid_layer:
            raise ValueError("Could not find a valid layer. "
                             "Check if base.logits or the mask result of base._mask_output() contains only zeros. "
                             "Check if requires_grad flag is true for the batch input and that no torch.no_grad statements effects gcam. "
                             "Check if the model has any convolution layers.")

        return layer, fmaps, grads

    def _extract_attentions(self, layer):
        fmaps = self._find(self.fmap_pool, layer)
        grads = self._find(self.grad_pool, layer)
        attention_map = self._generate_helper(fmaps, grads)
        return attention_map

    def _generate_helper(self, fmaps, grads):  # TODO: Make batch compatible
        weights = self._compute_grad_weights(grads)
        attention_map = torch.mul(fmaps, weights)
        #attention_map = attention_map[:, 0, ...].unsqueeze(1)
        B, _, *data_shape = attention_map.shape
        attention_map = attention_map.view(B, self.channels, -1, *data_shape)
        attention_map = torch.sum(attention_map, dim=2)  # TODO: mean or sum?
        attention_map = F.relu(attention_map)
        attention_map = self._normalize(attention_map)
        return attention_map

    def _normalize(self, attention_map):
        # TODO: Normalize over entire data or over every channel?
        B, C, *data_shape = attention_map.shape
        attention_map = attention_map.view(B, C, -1)
        attention_map_min = torch.min(attention_map, dim=2, keepdim=True)[0]
        attention_map_max = torch.max(attention_map, dim=2, keepdim=True)[0]
        attention_map -= attention_map_min
        attention_map /= (attention_map_max - attention_map_min)
        attention_map = attention_map.view(B, C, *data_shape)
        return attention_map

    def _check_hooks(self, layer):
        # TODO: Needs to be added to _BaseWrapper as other backends have also a generate method
        if not self.registered_hooks[layer][0] and not self.registered_hooks[layer][1]:
            raise ValueError("Neither forward hook nor backward hook did register to layer: " + str(layer))
        elif not self.registered_hooks[layer][0]:
            raise ValueError("Forward hook did not register to layer: " + str(layer))
        elif not self.registered_hooks[layer][1]:
            raise ValueError("Backward hook did not register to layer: " + str(layer))
