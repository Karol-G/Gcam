#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import OrderedDict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

def create_base_wrapper(base):
    class _BaseWrapper(base):
        """
        Please modify forward() and backward() according to your task.
        """

        def __init__(self, model, is_backward_ready=None):
            super(_BaseWrapper, self).__init__()
            self.device = next(model.parameters()).device
            self.model = model
            self.handlers = []  # a set of hook function handlers
            if is_backward_ready is None:
                self.is_backward_ready = self.model.is_backward_ready()
            else:
                self.is_backward_ready = is_backward_ready

        def _encode_one_hot(self, ids):
            one_hot = torch.zeros_like(self.logits).to(self.device)
            one_hot.scatter_(1, ids, 1.0)
            return one_hot

        def forward(self, image):
            """
            Simple classification
            """
            self.model.zero_grad()
            self.logits = self.model(image)
            return self.logits

        def backward(self, ids=None, output=None):
            """
            Class-specific backpropagation

            Either way works:
            1. self.logits.backward(gradient=one_hot, retain_graph=True)
            2. (self.logits * one_hot).sum().backward(retain_graph=True)
            """

            if output is not None:
                self.logits = output

            if self.is_backward_ready:
                self.logits.backward(gradient=self.logits, retain_graph=True)
            else:
                if ids is None:
                    ids = self.model.get_category_id_pos()
                # one_hot = self._encode_one_hot(ids)
                one_hot = torch.zeros_like(self.logits).to(self.device)
                for i in range(one_hot.shape[0]):
                    one_hot[i, ids[i]] = 1.0
                self.logits.backward(gradient=one_hot, retain_graph=True)

        def generate(self):
            raise NotImplementedError

        def remove_hook(self):
            """
            Remove all the forward/backward hook functions
            """
            for handle in self.handlers:
                handle.remove()
    return _BaseWrapper



def create_back_propagation(base):
    class BackPropagation(create_base_wrapper(base)):
        def forward(self, image):
            self.image = image.requires_grad_()
            return super(BackPropagation, self).forward(self.image)

        def generate(self):
            gradient = self.image.grad.clone()
            self.image.grad.zero_()
            return gradient
    return BackPropagation


def create_guided_back_propagation(base):
    class GuidedBackPropagation(create_back_propagation(base)):
        """
        "Striving for Simplicity: the All Convolutional Net"
        https://arxiv.org/pdf/1412.6806.pdf
        Look at Figure 1 on page 8.
        """

        def __init__(self, model):
            super(GuidedBackPropagation, self).__init__(model)

            def backward_hook(module, grad_in, grad_out):
                # Cut off negative gradients
                if isinstance(module, nn.ReLU):
                    return (torch.clamp(grad_in[0], min=0.0),)

            for module in self.model.named_modules():
                self.handlers.append(module[1].register_backward_hook(backward_hook))
    return GuidedBackPropagation


def create_deconvnet(base):
    class Deconvnet(create_back_propagation(base)):
        """
        "Striving for Simplicity: the All Convolutional Net"
        https://arxiv.org/pdf/1412.6806.pdf
        Look at Figure 1 on page 8.
        """

        def __init__(self, model):
            super(Deconvnet, self).__init__(model)

            def backward_hook(module, grad_in, grad_out):
                # Cut off negative gradients and ignore ReLU
                if isinstance(module, nn.ReLU):
                    return (torch.clamp(grad_out[0], min=0.0),)

            for module in self.model.named_modules():
                self.handlers.append(module[1].register_backward_hook(backward_hook))
    return Deconvnet

def detach_output(output):
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

def create_grad_cam(base):
    class GradCAM(create_base_wrapper(base)):
        """
        "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
        https://arxiv.org/pdf/1610.02391.pdf
        Look at Figure 2 on page 4
        """

        def __init__(self, model, target_layers=None, is_backward_ready=None):
            super(GradCAM, self).__init__(model, is_backward_ready)
            self.fmap_pool = OrderedDict()
            self.grad_pool = OrderedDict()
            self.module_names = {}
            self._target_layers = target_layers
            if target_layers == 'full' or target_layers == 'auto':
                target_layers = np.asarray(list(self.model.named_modules()))[:, 0]
            elif isinstance(target_layers, str):
                target_layers = [target_layers]
            self.target_layers = target_layers  # list
            self.registered_hooks = {}

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
                    self.registered_hooks[key][1] = True
                    # Save the gradients correspond to the featuremaps
                    self.grad_pool[key] = grad_out[0].detach()

                return backward_hook_

            # If any candidates are not specified, the hook is registered to all the layers.
            for name, module in self.model.named_modules():
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
            return F.adaptive_avg_pool2d(grads, 1)

        def forward(self, image, image_shape):
            #self.image_shape = image.shape[2:]
            self.image_shape = image_shape
            return super(GradCAM, self).forward(image)

        def select_highest_layer(self):
            fmap_list, weight_list = [], []
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
                        np.shape(fmaps) # Throws error without this line, I have no idea why...
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
                        weight_list.append(self._compute_grad_weights(grads))
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

            return layer, fmap_list, weight_list

        def generate_helper(self, fmaps, weights):
            attention_map = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
            attention_map = F.relu(attention_map)

            # print("attention_map.shape: ", attention_map.shape)
            #print("self.image_shape: ", self.image_shape)
            # attention_map = F.interpolate(
            #     attention_map, self.image_shape, mode="bilinear", align_corners=False
            # )
            #print("attention_map.shape: ", attention_map)
            B, C, H, W = attention_map.shape
            attention_map = attention_map.view(B, -1)
            attention_map -= attention_map.min(dim=1, keepdim=True)[0]
            attention_map /= attention_map.max(dim=1, keepdim=True)[0]
            attention_map = attention_map.view(B, C, H, W)

            return attention_map

        def extract_attentions(self, layer):
            fmaps = self._find(self.fmap_pool, layer)
            grads = self._find(self.grad_pool, layer)
            weights = self._compute_grad_weights(grads)
            gcam_tensor = self.generate_helper(fmaps, weights)
            attention_maps = []
            for i in range(self.logits.shape[0]):
                attention_map = gcam_tensor[i].unsqueeze(0)
                attention_map = attention_map.squeeze().cpu().numpy()
                attention_maps.append(attention_map)
            return attention_maps

        def generate(self):
            if self._target_layers == "auto":
                layer, fmaps, weights = self.select_highest_layer()
                self.check_hooks(layer)
                attention_maps = []
                for i in range(self.logits.shape[0]):
                    attention_map = self.generate_helper(fmaps[i].unsqueeze(0), weights[i].unsqueeze(0))
                    attention_map = attention_map.squeeze().cpu().numpy()
                    attention_maps.append(attention_map)
                attention_maps = {layer: attention_maps}
            else:
                attention_maps = {}
                for layer in self.target_layers:
                    self.check_hooks(layer)
                    attention_maps_tmp = self.extract_attentions(str(layer))
                    attention_maps[layer] = attention_maps_tmp
            return attention_maps

        def check_hooks(self, layer):
            # TODO: Needs to be added to _BaseWrapper as other backends have also a generate method
            if not self.registered_hooks[layer][0] and not self.registered_hooks[layer][1]:
                raise ValueError("Neither forward hook nor backward hook did register to layer: " + str(layer))
            elif not self.registered_hooks[layer][0]:
                raise ValueError("Forward hook did not register to layer: " + str(layer))
            elif not self.registered_hooks[layer][1]:
                raise ValueError("Backward hook did not register to layer: " + str(layer))

    return GradCAM


def occlusion_sensitivity(
    model, images, ids, mean=None, patch=35, stride=1, n_batches=128
):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17
    
    Originally proposed in:
    "Visualizing and Understanding Convolutional Networks"
    https://arxiv.org/abs/1311.2901
    """

    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(images).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        batch_images = []
        batch_ids = []
        for grid_h, grid_w in anchors[i : i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_ids.append(ids)
        batch_images = torch.cat(batch_images, dim=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        scores = model(batch_images).detach().gather(1, batch_ids)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps
