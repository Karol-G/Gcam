import torch
from pathlib import Path
import types
import pickle
import pandas as pd
from gcam import gcam_utils
from gcam.backends.guided_backpropagation import GuidedBackPropagation
from gcam.backends.grad_cam import GradCAM
from gcam.backends.guided_grad_cam import GuidedGradCam
from gcam.backends.grad_cam_pp import GradCamPP
from gcam import score_utils
from collections import defaultdict
from torch.nn import functional as F
import numpy as np

# TODO: Set requirements in setup.py

def inject(model, output_dir=None, backend='gcam', layer='auto', input_key=None, mask_key=None, postprocessor=None,
           retain_graph=False, save_scores=False, save_maps=False, save_pickle=False, evaluate=False, metric='wioa',
           return_score=False, threshold=0.3, registered_only=False, label=None, channels='default', data_shape='default'):

    if _already_injected(model):
        return

    # torch.backends.cudnn.enabled = False # TODO: out of memory
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.eval()
    # Save the original forward of the model
    # This forward will be called by the backend, so if someone writes a new backend they only need to call model.model_forward and not model.gcam_dict['model_forward']
    setattr(model, 'model_forward', model.forward)

    # Save every other attribute in a dict which is added to the model attributes
    # It is ugly but it avoids name conflicts
    gcam_dict = {}

    gcam_dict['output_dir'] = output_dir
    gcam_dict['layer'] = layer
    gcam_dict['input_key'] = input_key
    gcam_dict['mask_key'] = mask_key
    gcam_dict['counter'] = 0
    gcam_dict['save_scores'] = save_scores
    gcam_dict['save_maps'] = save_maps
    gcam_dict['save_pickle'] = save_pickle
    gcam_dict['evaluate'] = evaluate
    gcam_dict['metric'] = metric
    gcam_dict['return_score'] = return_score
    gcam_dict['_replace_output'] = False
    gcam_dict['threshold'] = threshold
    gcam_dict['label'] = label
    gcam_dict['channels'] = channels
    gcam_dict['data_shape'] = data_shape
    gcam_dict['pickle_maps'] = []
    gcam_dict['scores'] = defaultdict(list)
    gcam_dict['current_attention_map'] = None
    gcam_dict['current_layer'] = None
    gcam_dict['device'] = next(model.parameters()).device
    setattr(model, 'gcam_dict', gcam_dict)

    if output_dir is None and (save_scores is not None or save_maps is not None or save_pickle is not None):
        raise ValueError("output_dir needs to be set if save_scores, save_maps or save_pickle is set to true")

    # Append methods methods to the model
    model.get_layers = types.MethodType(get_layers, model)
    model.get_attention_map = types.MethodType(get_attention_map, model)
    model.save_attention_map = types.MethodType(save_attention_map, model)
    model.replace_output = types.MethodType(replace_output, model)
    model.dump = types.MethodType(dump, model)
    model.forward = types.MethodType(forward, model)

    model._assign_backend = types.MethodType(_assign_backend, model)
    model._unpack_batch = types.MethodType(_unpack_batch, model)
    model._process_attention_maps = types.MethodType(_process_attention_maps, model)
    model._save_file = types.MethodType(_save_attention_map, model)
    model._comp_score = types.MethodType(_comp_score, model)
    model._comp_mean_score = types.MethodType(_comp_mean_score, model)
    model._scores2csv = types.MethodType(_scores2csv, model)
    model._replace_output = types.MethodType(_replace_output, model)
    model._extract_metadata = types.MethodType(_extract_metadata, model)

    model_backend, heatmap = _assign_backend(backend, model, layer, postprocessor, retain_graph, registered_only)
    gcam_dict['model_backend'] = model_backend
    gcam_dict['heatmap'] = heatmap


def get_layers(self, reverse=False):
    return self.gcam_dict['model_backend'].layers(reverse)

def get_attention_map(self):
    return self.gcam_dict['current_attention_map']

def save_attention_map(self, attention_map):
    gcam_utils.save_attention_map(filename=self.gcam_dict['output_dir'] + "/" + self.gcam_dict['current_layer'] + "/attention_map_" +
                                           str(self.gcam_dict['counter']), attention_map=attention_map, heatmap=self.gcam_dict['heatmap'])
    self.gcam_dict['counter'] += 1

def replace_output(self, replace):
    self.gcam_dict['_replace_output'] = replace

def dump(self, show=True):
    if self.gcam_dict['save_pickle']:
        with open(self.gcam_dict['output_dir'] + '/attention_maps.pkl', 'wb') as handle:  # TODO: Save every 1GB
            pickle.dump(self.gcam_dict['pickle_maps'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    mean_scores = self._comp_mean_score(show)
    if self.gcam_dict['save_scores']:
        self._scores2csv(mean_scores)

def forward(self, batch, label=None, mask=None):
    # print("-------------------------- FORWARD GCAM HOOK --------------------------")
    with torch.enable_grad():
        batch_size, channels, data_shape = self._unpack_batch(batch)
        output = self.gcam_dict['model_backend'].forward(batch)
        self.gcam_dict['model_backend'].backward(label=label)
        attention_map = self.gcam_dict['model_backend'].generate()
        if len(attention_map.keys()) == 1:
            self.gcam_dict['current_attention_map'] = attention_map[list(attention_map.keys())[0]]
            self.gcam_dict['current_layer'] = list(attention_map.keys())[0]
        scores = self._process_attention_maps(attention_map, batch, mask, batch_size, channels)
        output = self._replace_output(output, attention_map, data_shape)
        if self.gcam_dict['return_score']:
            return output, scores
        else:
            return output

def _already_injected(model):
    try:  # try/except is faster than hasattr, if inject method is called repeatedly
        model.gcam_dict  # Check if attribute exists
        return True
    except AttributeError:
        return False

def _assign_backend(backend, model, target_layers, postprocessor, retain_graph, registered_only):
    if backend == "gbp":
        return GuidedBackPropagation(model=model, postprocessor=postprocessor, retain_graph=retain_graph), False
    elif backend == "gcam":
        return GradCAM(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph, registered_only=registered_only), True
    elif backend == "ggcam":
        return GuidedGradCam(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph), False
    elif backend == "gcampp":
        return GradCamPP(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph), True
    else:
        raise ValueError("Backend does not exist")

def _unpack_batch(self, batch):
    if self.gcam_dict['input_key'] is None:
        batch_size, channels, data_shape = self._extract_metadata(batch)
    else:
        batch_size, channels, data_shape = self._extract_metadata(batch[self.gcam_dict['input_key']])
    return batch_size, channels, data_shape

def _process_attention_maps(self, attention_map, batch, mask, batch_size, channels):
    batch_scores = defaultdict(list) if self.gcam_dict['evaluate'] else None
    for layer_name in attention_map.keys():
        layer_output_dir = None
        if self.gcam_dict['output_dir'] is not None:
            if layer_name == "":
                layer_output_dir = self.gcam_dict['output_dir']
            else:
                layer_output_dir = self.gcam_dict['output_dir'] + "/" + layer_name
            Path(layer_output_dir).mkdir(parents=True, exist_ok=True)
        for j in range(batch_size):
            for k in range(channels):
                attention_map_single = attention_map[layer_name][j][k]
                self._save_file(attention_map_single, layer_output_dir)
                if self.gcam_dict['evaluate']:
                    if mask is None:
                        raise ValueError("Mask cannot be none in evaluation mode")
                    score = self._comp_score(attention_map_single, batch, mask[j][k].squeeze())
                    batch_scores[layer_name].append(score)
                    self.gcam_dict['scores'][layer_name].append(score)
    return batch_scores

def _save_attention_map(self, attention_map, layer_output_dir):
    if self.gcam_dict['save_pickle']:
        self.gcam_dict['pickle_maps'].append(attention_map)
    if self.gcam_dict['save_maps']:
        gcam_utils.save_attention_map(filename=layer_output_dir + "/attention_map_" + str(self.gcam_dict['counter']), attention_map=attention_map, heatmap=self.gcam_dict['heatmap'])
        self.gcam_dict['counter'] += 1

def _comp_score(self, attention_map, batch, mask):  # TODO: Not multiclass compatible, maybe multiclass parameter in init?
    if self.gcam_dict['mask_key'] is not None:
        mask = batch[self.gcam_dict['mask_key']]
    elif mask is None:
        raise ValueError("Either mask_key during initialization or mask during forward needs to be set")
    return score_utils.comp_score(attention_map, mask, self.gcam_dict['metric'], self.gcam_dict['threshold'])

def _comp_mean_score(self, show):
    mean_scores = defaultdict(float)
    for layer_name in self.gcam_dict['scores'].keys():
        mean_score = np.mean(self.gcam_dict['scores'][layer_name])
        mean_scores[layer_name] = mean_score
        if show:
            print("Layer: {}, mean score: {}".format(layer_name, mean_score))
    return mean_scores

def _scores2csv(self, mean_scores):
    df = pd.DataFrame.from_dict(self.gcam_dict['scores'])
    new_entry = pd.DataFrame([mean_scores.values()], columns=mean_scores.keys())
    score_ids = list(range(len(df)))
    score_ids = list(map(str, score_ids))
    score_ids.append("Mean")
    df = df.append(new_entry)
    df.insert(0, "Layer", score_ids, True)
    df.to_csv(self.gcam_dict['output_dir'] + "/scores.csv", index=False)

def _replace_output(self, output, attention_map, data_shape):
    if self.gcam_dict['_replace_output']:
        if len(attention_map.keys()) == 1:
            if self.gcam_dict['data_shape'] == 'default':
                output_shape = data_shape
            else:
                output_shape = self.gcam_dict['data_shape']
            output = torch.tensor(self.gcam_dict['current_attention_map']).to(str(self.gcam_dict['device']))  # TODO: Test with 2D
            output = gcam_utils.interpolate(output, output_shape)
        else:
            raise ValueError("Not possible to replace output when layer is 'full', only with 'auto' or a manually set layer")
    return output

def _extract_metadata(self, batch):
    batch_size = batch.shape[0]
    if self.gcam_dict['channels'] == 'default':
        channels = batch.shape[1]
    else:
        channels = self.gcam_dict['channels']
    if self.gcam_dict['data_shape'] == 'default':
        data_shape = batch.shape[2:]
    else:
        data_shape = self.model.gcam_dict['data_shape']
    return batch_size, channels, data_shape