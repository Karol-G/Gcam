import torch
from pathlib import Path
import types
import pickle
from gcam import gcam_utils
from gcam.backends.guided_backpropagation import GuidedBackPropagation
from gcam.backends.grad_cam import GradCAM
from gcam.backends.guided_grad_cam import GuidedGradCam
from gcam.backends.grad_cam_pp import GradCamPP
from collections import defaultdict
from gcam.evaluation.evaluator import Evaluator
import copy
import numpy as np

def inject(model, output_dir=None, backend='gcam', layer='auto', channels='default', data_shape='default', postprocessor=None, label=None,
           save_maps=False, save_pickle=False, save_scores=False, evaluate=False, metric='wioa', threshold='otsu', retain_graph=False,
           return_score=False, replace=False, cudnn=True, test_batch=None, enabled=True):
    """
    Injects a model with gcam functionality to extract attention maps from it. The model can be used as usual.
    Whenever model(input) or model.forward(input) is called gcam will extract the corresponding attention maps.
    Args:
        model: A CNN-based model that inherits from torch.nn.Module
        output_dir: The directory to save any results to
        backend: One of the implemented visualization backends.

                'gbp': Guided-Backpropagation

                'gcam': Grad-Cam

                'ggcam': Guided-Grad-Cam

                'gcampp': Grad-Cam++

        layer: One or multiple layer names of the model from which attention maps will be extracted.

                'auto': Selects the last layer from which attention maps can be extracted.

                'full': Selects every layer from which attention maps can be extracted.

                (layer name): A layer name of the model as string.

                [(layer name 1), (layer name 2), ...]: A list of layer names of the model as string.

            Note: Guided-Backpropagation ignores this parameter.

        channels: The number of channels the attention maps should have. Some models (e.g. segmentation models) use the channel dimension for class discrimination. For these models the number of channels should corresponds to the number of classes.

                'default': The number of channels of the current input data.

        data_shape: The shape of the resulting attention maps. The given shape should exclude batch and channel dimension.

                'default': The shape of the current input data, excluding batch and channel dimension.

        postprocessor: The postprocessor is applied on the model output from calling forward which is then passed to the class discriminator. It converts the raw logit output from the model to a usable form for the class discriminator. The postprocessor is only applied internally, the final output given to the user is not effected.

                None: No postprocessing is applied.

                'sigmoid': Applies the sigmoid function.

                'softmax': Applies softmax function.

                (A function): Applies a given function to the output.

        label: A class discriminator that creates a mask on the postprocessed output. Only the non masked logits are backwarded through the model.

                Example: label=lambda x: 0.5 < x

        save_maps: If the attention maps should be saved sorted by layer in the output_dir.

        save_pickle: If the attention maps should be saved as a pickle file in the output_dir.

        save_scores: If the evaluation scores should be saved as an excel file in the output_dir.

        evaluate: If the attention maps should be evaluated. This requires a corresponding mask when calling model.forward().

        metric: An evaluation metric for comparing the attention map with the mask.

                'wioa': Weighted intersection over attention. Most suited for classification.

                'ioa': Intersection over attention.

                'iou': Intersection over union. Not suited for classification.

                (A function): An evaluation function.

        threshold: A threshold used during evaluation for ignoring low attention. Most models have low amounts of attention everywhere in an attention map due to the nature of CNN-based models. The threshold can be used to ignore these low amounts if wanted.

                'otsu': Uses the otsu algorithm to determine a threshold.

                (float): A value between 0 and 1 that is used as threshold.

        retain_graph: If the computation graph should be retained or not.

        return_score: If the evaluation evaluation of the current input should be returned in addition to the model output.

        replace: If the model output should be replaced with the extracted attention map.

        cudnn: If cudnn should be disabled. Some models (e.g. LSTMs) crash when using gcam with enabled cudnn.

        test_batch: A test input. This allows gcam to determine compatible layers.

        enabled: If gcam should be enabled.

    Returns: A shallow copy of the model injected with gcam functionality.

    """

    if _already_injected(model):
        return

    if not cudnn:
        torch.backends.cudnn.enabled = False

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_clone = copy.copy(model)
    model_clone.eval()
    # Save the original forward of the model
    # This forward will be called by the backend, so if someone writes a new backend they only need to call model.model_forward and not model.gcam_dict['model_forward']
    setattr(model_clone, 'model_forward', model_clone.forward)

    # Save every other attribute in a dict which is added to the model attributes
    # It is ugly but it avoids name conflicts
    gcam_dict = {}

    gcam_dict['output_dir'] = output_dir
    gcam_dict['layer'] = layer
    gcam_dict['counter'] = 0
    gcam_dict['save_scores'] = save_scores
    gcam_dict['save_maps'] = save_maps
    gcam_dict['save_pickle'] = save_pickle
    gcam_dict['evaluate'] = evaluate
    gcam_dict['metric'] = metric
    gcam_dict['return_score'] = return_score
    gcam_dict['_replace_output'] = replace
    gcam_dict['threshold'] = threshold
    gcam_dict['label'] = label
    gcam_dict['channels'] = channels
    gcam_dict['data_shape'] = data_shape
    gcam_dict['pickle_maps'] = []
    if evaluate:
        gcam_dict['Evaluator'] = Evaluator(output_dir + "/", metric=metric, threshold=threshold, layer_ordering=gcam_utils.get_layers(model_clone))
    gcam_dict['current_attention_map'] = None
    gcam_dict['current_layer'] = None
    gcam_dict['device'] = next(model_clone.parameters()).device
    gcam_dict['tested'] = False
    gcam_dict['enabled'] = enabled
    setattr(model_clone, 'gcam_dict', gcam_dict)

    if output_dir is None and (save_scores is not None or save_maps is not None or save_pickle is not None or evaluate):
        raise ValueError("output_dir needs to be set if save_scores, save_maps, save_pickle or evaluate is set to true")

    # Append methods methods to the model
    model_clone.get_layers = types.MethodType(get_layers, model_clone)
    model_clone.get_attention_map = types.MethodType(get_attention_map, model_clone)
    model_clone.save_attention_map = types.MethodType(save_attention_map, model_clone)
    model_clone.replace_output = types.MethodType(replace_output, model_clone)
    model_clone.dump = types.MethodType(dump, model_clone)
    model_clone.forward = types.MethodType(forward, model_clone)
    model_clone.enable_gcam = types.MethodType(enable_gcam, model_clone)
    model_clone.disable_gcam = types.MethodType(disable_gcam, model_clone)
    model_clone.test_run = types.MethodType(test_run, model_clone)

    model_clone._assign_backend = types.MethodType(_assign_backend, model_clone)
    model_clone._process_attention_maps = types.MethodType(_process_attention_maps, model_clone)
    model_clone._save_attention_map = types.MethodType(_save_attention_map, model_clone)
    model_clone._replace_output = types.MethodType(_replace_output, model_clone)
    model_clone._extract_metadata = types.MethodType(_extract_metadata, model_clone)

    model_backend, heatmap = _assign_backend(backend, model_clone, layer, postprocessor, retain_graph)
    gcam_dict['model_backend'] = model_backend
    gcam_dict['heatmap'] = heatmap

    model_clone.test_run(test_batch)

    return model_clone

def get_layers(self, reverse=False):
    """Returns the layers of the model. Optionally reverses the order of the layers."""
    return self.gcam_dict['model_backend'].layers(reverse)

def get_attention_map(self):
    """Returns the current attention map."""
    return self.gcam_dict['current_attention_map']

def save_attention_map(self, attention_map):
    """Saves an attention map."""
    gcam_utils.save_attention_map(filename=self.gcam_dict['output_dir'] + "/" + self.gcam_dict['current_layer'] + "/attention_map_" +
                                           str(self.gcam_dict['counter']), attention_map=attention_map, heatmap=self.gcam_dict['heatmap'])
    self.gcam_dict['counter'] += 1

def replace_output(self, replace):
    """If the output should be replaced with the corresponiding attention map."""
    self.gcam_dict['_replace_output'] = replace

def dump(self):
    """Saves all of the collected data to the output directory."""
    if self.gcam_dict['save_pickle']:
        with open(self.gcam_dict['output_dir'] + '/attention_maps.pkl', 'wb') as handle:  # TODO: Save every 1GB
            pickle.dump(self.gcam_dict['pickle_maps'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    if self.gcam_dict['save_scores']:
        self.gcam_dict['Evaluator'].dump()

def forward(self, batch, label=None, mask=None):
    """
    Generates attention maps for a given batch input.
    Args:
        batch: An input batch of shape (BxCxHxW) or (BxCxDxHxW).
        label: A class label (int) or a class discriminator function to different attention maps for every class.
        mask: A ground truth mask corresponding to the input batch. Only needed when evaluate is set to true.

    Returns: Either the normal output of the model or an attention map.

    """
    if self.gcam_dict['enabled']:
        if self.gcam_dict['layer'] == 'full' and not self.gcam_dict['tested']:
            raise ValueError("Layer mode 'full' requires a test run either during injection or by calling test_run() afterwards")
        with torch.enable_grad():
            output = self.gcam_dict['model_backend'].forward(batch)
            batch_size, channels, data_shape = self._extract_metadata(batch, output)
            self.gcam_dict['model_backend'].backward(label=label)
            attention_map = self.gcam_dict['model_backend'].generate()
            if attention_map:
                if len(attention_map.keys()) == 1:
                    self.gcam_dict['current_attention_map'] = attention_map[list(attention_map.keys())[0]]
                    self.gcam_dict['current_layer'] = list(attention_map.keys())[0]
                scores = self._process_attention_maps(attention_map, mask, batch_size, channels)
                output = self._replace_output(output, attention_map, data_shape)
            else:  # If no attention maps could be extracted
                self.gcam_dict['current_attention_map'] = None
                self.gcam_dict['current_layer'] = None
                scores = None
                if self.gcam_dict['_replace_output']:
                    raise ValueError("Unable to extract any attention maps")
            self.gcam_dict['counter'] += 1
            if self.gcam_dict['return_score']:
                return output, scores
            else:
                return output
    else:
        return self.model_forward(batch)

def test_run(self, batch):
    """Performs a test run. This allows gcam to determine for which layers it can generate attention maps."""
    registered_hooks = []
    if batch is not None and not self.gcam_dict['tested']:
        with torch.enable_grad():
            output = self.gcam_dict['model_backend'].forward(batch)
            self.gcam_dict['model_backend'].backward()
            registered_hooks = self.gcam_dict['model_backend'].get_registered_hooks()
        self.gcam_dict['tested'] = True
        print("Successfully registered to the following layers: ", registered_hooks)
        if self.gcam_dict['output_dir'] is not None:
            np.savetxt(self.gcam_dict['output_dir'] + '/registered_layers.txt', np.asarray(registered_hooks).astype(str), fmt="%s")
    return registered_hooks

def disable_gcam(self):
    """Disables gcam."""
    self.gcam_dict['enabled'] = False

def enable_gcam(self):
    """Enables gcam."""
    self.gcam_dict['enabled'] = True

def _already_injected(model):
    """Checks if the model is already injected with gcam."""
    try:  # try/except is faster than hasattr, if inject method is called repeatedly
        model.gcam_dict  # Check if attribute exists
        return True
    except AttributeError:
        return False

def _assign_backend(backend, model, target_layers, postprocessor, retain_graph):
    """Assigns a chosen backend."""
    if backend == "gbp":
        return GuidedBackPropagation(model=model, postprocessor=postprocessor, retain_graph=retain_graph), False
    elif backend == "gcam":
        return GradCAM(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph), True
    elif backend == "ggcam":
        return GuidedGradCam(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph), False
    elif backend == "gcampp":
        return GradCamPP(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph), True
    else:
        raise ValueError("Backend does not exist")

def _process_attention_maps(self, attention_map, mask, batch_size, channels):
    """Handles all the stuff after the attention map has been generated. Like creating dictionaries, saving the attention map and doing the evaluation."""
    batch_scores = defaultdict(list) if self.gcam_dict['evaluate'] else None
    for layer_name in attention_map.keys():
        layer_output_dir = None
        if self.gcam_dict['output_dir'] is not None and self.gcam_dict['save_maps']:
            if layer_name == "":
                layer_output_dir = self.gcam_dict['output_dir']
            else:
                layer_output_dir = self.gcam_dict['output_dir'] + "/" + layer_name
            Path(layer_output_dir).mkdir(parents=True, exist_ok=True)
        for j in range(batch_size):
            for k in range(channels):
                attention_map_single = attention_map[layer_name][j][k]
                self._save_attention_map(attention_map_single, layer_output_dir, j, k)
                if self.gcam_dict['evaluate']:
                    if mask is None:
                        raise ValueError("Mask cannot be none in evaluation mode")
                    score = self.gcam_dict['Evaluator'].comp_score(attention_map_single, mask[j][k].squeeze(), layer=layer_name, class_label=k)
                    batch_scores[layer_name].append(score)
    return batch_scores

def _save_attention_map(self, attention_map, layer_output_dir, j, k):
    """Internal method for saving saving an attention map."""
    if self.gcam_dict['save_pickle']:
        self.gcam_dict['pickle_maps'].append(attention_map)
    if self.gcam_dict['save_maps']:
        gcam_utils.save_attention_map(filename=layer_output_dir + "/attention_map_" + str(self.gcam_dict['counter']) + "_" + str(j) + "_" + str(k), attention_map=attention_map, heatmap=self.gcam_dict['heatmap'])

def _replace_output(self, output, attention_map, data_shape):
    """Replaces the model output with the current attention map."""
    if self.gcam_dict['_replace_output']:
        if len(attention_map.keys()) == 1:
            output = torch.tensor(self.gcam_dict['current_attention_map']).to(str(self.gcam_dict['device']))
            output = gcam_utils.interpolate(output, data_shape)
        else:
            raise ValueError("Not possible to replace output when layer is 'full', only with 'auto' or a manually set layer")
    return output

def _extract_metadata(self, input, output):  # TODO: Does not work for classification output (shape: (1, 1000))
    """Extracts metadata like batch size, number of channels and the data shape from the input batch."""
    output_batch_size = output.shape[0]
    if self.gcam_dict['channels'] == 'default':
        output_channels = output.shape[1]
    else:
        output_channels = self.gcam_dict['channels']
    if self.gcam_dict['data_shape'] == 'default':
        output_shape = output.shape[2:]
    else:
        output_shape = self.model.gcam_dict['data_shape']
    return output_batch_size, output_channels, output_shape