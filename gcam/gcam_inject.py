import torch
from pathlib import Path
import types
import pickle
import pandas as pd
from gcam import gcam_utils
from gcam.backends.guided_backpropagation import create_guided_back_propagation
from gcam.backends.grad_cam import create_grad_cam
from gcam.backends.guided_grad_cam import create_guided_grad_cam
from gcam.backends.grad_cam_pp import create_grad_cam_pp
from gcam import score_utils
from collections import defaultdict
import copy
import numpy as np

# TODO: Set requirements in setup.py

def inject(model, output_dir=None, backend="gcam", layer='auto', input_key=None, mask_key=None, postprocessor=None,
           retain_graph=False, dim=2, save_scores=False, save_maps=False, save_pickle=False, evaluate=False, metric="wioa",
           return_score=False, threshold=0.3, registered_only=False):
    # torch.backends.cudnn.enabled = False # TODO: out of memory
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.eval()
    model_backend = copy.deepcopy(model)
    model_backend, heatmap = _assign_backend(backend, model_backend, layer, postprocessor, retain_graph, dim, registered_only)

    setattr(model, 'output_dir', output_dir)
    setattr(model, 'layer', layer)
    setattr(model, 'input_key', input_key)
    setattr(model, 'mask_key', mask_key)
    setattr(model, 'model_backend', model_backend)
    setattr(model, 'heatmap', heatmap)
    setattr(model, 'counter', 0)
    setattr(model, 'dim', dim)
    setattr(model, 'save_scores', save_scores)
    setattr(model, 'save_maps', save_maps)
    setattr(model, 'save_pickle', save_pickle)
    setattr(model, 'evaluate', evaluate)
    setattr(model, 'metric', metric)
    setattr(model, 'return_score', return_score)
    setattr(model, '_replace_output', False)
    setattr(model, 'threshold', threshold)
    setattr(model, 'pickle_maps', [])
    setattr(model, 'scores', defaultdict(list))
    setattr(model, 'current_attention_map', None)
    setattr(model, 'current_layer', None)
    setattr(model, 'device', next(model.parameters()).device)

    if output_dir is None and (save_scores is not None or save_maps is not None or save_pickle is not None):
        raise AttributeError("output_dir needs to be set if save_scores, save_maps or save_pickle is set to true")

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


def get_layers(self, reverse=False):
    return self.model_backend.layers(reverse)

def get_attention_map(self):
    return self.current_attention_map

def save_attention_map(self, attention_map):
    gcam_utils.save_attention_map(filename=self.output_dir + "/" + self.current_layer + "/attention_map_" + str(self.counter), attention_map=attention_map, heatmap=self.heatmap, dim=self.dim)
    self.counter += 1

def replace_output(self, replace):
    self._replace_output = replace

def dump(self, show=True):
    if self.save_pickle:
        with open(self.output_dir + '/attention_maps.pkl', 'wb') as handle:  # TODO: Save every 1GB
            pickle.dump(self.pickle_maps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mean_scores = self._comp_mean_score(show)
    if self.save_scores:
        self._scores2csv(mean_scores)

def forward(self, batch, label=None, mask=None):
    # print("-------------------------- FORWARD GCAM HOOK --------------------------")
    self.model_backend.model.eval()
    with torch.enable_grad():
        batch_size, data_shape = self._unpack_batch(batch)
        output = self.model_backend.forward(batch, data_shape)
        self.model_backend.backward(output=output, label=label)  # TODO: Check if I can remove output
        attention_map = self.model_backend.generate()
        if len(attention_map.keys()) == 1:
            self.current_attention_map = attention_map[list(attention_map.keys())[0]][0]
            #self.current_attention_map = gradcam_utils.normalize(self.current_attention_map)*255.0
            #self.current_attention_map = torch.tensor(self.current_attention_map).unsqueeze(0).unsqueeze(0).to(str(self.device))
            self.current_layer = list(attention_map.keys())[0]
        scores = self._process_attention_maps(attention_map, batch, mask, batch_size)
        if self._replace_output:
            if len(attention_map.keys()) == 1:
                output = torch.tensor(self.current_attention_map).unsqueeze(0).unsqueeze(0).to(str(self.device))
            else:
                raise RuntimeError("Not possible to replace output when layer is 'full', only with 'auto' or a manually set layer")
        if self.return_score:
            return output, scores
        else:
            #output = gradcam_utils.normalize(output) * 255.0
            return output

def _assign_backend(backend, model, target_layers, postprocessor, retain_graph, dim, registered_only):
    if backend == "gbp":
        return create_guided_back_propagation(object)(model=model, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim), False
    elif backend == "gcam":
        return create_grad_cam(object)(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim, registered_only=registered_only), True
    elif backend == "ggcam":
        return create_guided_grad_cam(object)(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim), False
    elif backend == "gcampp":
        return create_grad_cam_pp(object)(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim), True
    else:
        raise TypeError("Backend does not exist")

def _unpack_batch(self, batch):
    if self.input_key is None:
        data_shape = batch.shape[-self.dim:]
        batch_size = batch.shape[0]
    else:
        data_shape = batch[self.input_key].shape[-self.dim:]
        batch_size = batch[self.input_key].shape[0]
    return batch_size, data_shape

def _process_attention_maps(self, attention_map, batch, mask, batch_size):
    batch_scores = defaultdict(list) if self.evaluate else None
    for layer_name in attention_map.keys():
        layer_output_dir = None
        if self.output_dir is not None:
            if layer_name == "":
                layer_output_dir = self.output_dir
            else:
                layer_output_dir = self.output_dir + "/" + layer_name
            Path(layer_output_dir).mkdir(parents=True, exist_ok=True)
        for j in range(batch_size):
            attention_map_j = attention_map[layer_name][j]
            self._save_file(attention_map_j, layer_output_dir)
            if self.evaluate:
                score = self._comp_score(attention_map_j, batch, mask[j].squeeze())
                batch_scores[layer_name].append(score)
                self.scores[layer_name].append(score)
    return batch_scores

def _save_attention_map(self, attention_map, layer_output_dir):
    if self.save_pickle:
        self.pickle_maps.append(attention_map)
    if self.save_maps:
        gcam_utils.save_attention_map(filename=layer_output_dir + "/attention_map_" + str(self.counter), attention_map=attention_map, heatmap=self.heatmap, dim=self.dim)
        self.counter += 1

def _comp_score(self, attention_map, batch, mask):  # TODO: Not multiclass compatible, maybe multiclass parameter in init?
    if self.mask_key is not None:
        mask = batch[self.mask_key]
    elif mask is None:
        raise AttributeError("Either mask_key during initialization or mask during forward needs to be set")
    return score_utils.comp_score(attention_map, mask, self.metric, self.threshold)

def _comp_mean_score(self, show):
    mean_scores = defaultdict(float)
    for layer_name in self.scores.keys():
        mean_score = np.mean(self.scores[layer_name])
        mean_scores[layer_name] = mean_score
        if show:
            print("Layer: {}, mean score: {}".format(layer_name, mean_score))
    return mean_scores

def _scores2csv(self, mean_scores):
    df = pd.DataFrame.from_dict(self.scores)
    new_entry = pd.DataFrame([mean_scores.values()], columns=mean_scores.keys())
    score_ids = list(range(len(df)))
    score_ids = list(map(str, score_ids))
    score_ids.append("Mean")
    df = df.append(new_entry)
    df.insert(0, "Layer", score_ids, True)
    df.to_csv(self.output_dir + "/scores.csv", index=False)