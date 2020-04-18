import torch
from pathlib import Path
import types
import functools
import pickle
import pandas as pd
from gcam.gradcam_utils import *
from gcam.backends.guided_backpropagation import create_guided_back_propagation
from gcam.backends.grad_cam import create_grad_cam
from gcam.backends.guided_grad_cam import create_guided_grad_cam
from gcam.backends.grad_cam_pp import create_grad_cam_pp
from gcam import score_metrics
from collections import defaultdict

# TODO: Set requirements in setup.py

class Gcam():
    def __init__(self, model, output_dir=None, backend="gcam", layer='auto', input_key=None, mask_key=None, postprocessor=None,
                 retain_graph=False, dim=2, save_scores=False, save_maps=False, save_pickle=False, evaluate=False, metric="ioa", return_score=False, threshold=0.3):
        super(Gcam, self).__init__()
        self.__dict__ = model.__dict__.copy()
        # torch.backends.cudnn.enabled = False # TODO: out of memory
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.layer = layer
        self.input_key = input_key
        self.mask_key = mask_key
        self.model = model
        self.model.eval()
        self.model_backend, self.heatmap = self._assign_backend(backend, self.model, self.layer, postprocessor, retain_graph, dim)
        self.backend = backend
        self.counter = 0
        self.dim = dim
        self.save_scores = save_scores
        self.save_maps = save_maps
        self.save_pickle = save_pickle
        self.evaluate = evaluate
        self.metric = metric
        self.return_score = return_score
        self._replace_output = False
        self.threshold = threshold
        self.pickle_maps = []
        self.scores = defaultdict(list)
        self.current_attention_map = None
        self.current_layer = None
        self.device = next(self.model.parameters()).device

        if self.output_dir is None and (self.save_scores is not None or self.save_maps is not None or self.save_pickle is not None):
            raise AttributeError("output_dir needs to be set if save_log, save_maps or save_pickle is set to true")
        # print("--------------------SUPER TEST")

    def get_layers(self, reverse=False):
        return self.model_backend.layers(reverse)

    def get_attention_map(self):
        return self.current_attention_map

    def save_attention_map(self, attention_map):
        save_attention_map(filename=self.output_dir + "/" + self.current_layer + "/attention_map_" + str(self.counter), attention_map=attention_map, heatmap=self.heatmap, dim=self.dim)
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

    def __call__(self, batch, label=None, mask=None):
        return self.forward(batch, label, mask)

    def forward(self, batch, label=None, mask=None):
        # print("-------------------------- FORWARD GCAM HOOK --------------------------")
        with torch.enable_grad():
            batch_size, data_shape = self._unpack_batch(batch)
            output = self.model_backend.forward(batch, data_shape)
            self.model_backend.backward(output=output, label=label)  # TODO: Check if I can remove output
            attention_map = self.model_backend.generate()
            if len(attention_map.keys()) == 1:
                self.current_attention_map = torch.tensor(attention_map[list(attention_map.keys())[0]][0]).unsqueeze(0).unsqueeze(0).to(str(self.device))
                self.current_layer = list(attention_map.keys())[0]
            scores = self._process_attention_maps(attention_map, batch, mask, batch_size)
            if self._replace_output:
                if len(attention_map.keys()) == 1:
                    output = self.current_attention_map
                else:
                    raise RuntimeError("Not possible to replace output when layer is 'full', only with 'auto' or a manually set layer")
            if self.return_score:
                return output, scores
            else:
                return output

    # TODO: If GcamHook.predict3D(...) gets called, it will be forwarded to model.predict3D(...) and then model.forward(...) wil be called not GcamHook.forward(...)
    # TODO: https://stackoverflow.com/questions/243836/how-to-copy-all-properties-of-an-object-to-another-object-in-python
    # TODO: Maybe replace copy with deepcopy
    # TODO: https://stackoverflow.com/questions/26467564/how-to-copy-all-attributes-of-one-python-object-to-another/26467767

    # TODO: Retain graph only with normal backward afterwards, otherwise out of memory
    # TODO: Save list if memory size is 1GB
    # TODO: https://stackoverflow.com/questions/20771470/list-memory-usage

    def _assign_backend(self, backend, model, target_layers, postprocessor, retain_graph, dim):
        if backend == "gbp":
            return create_guided_back_propagation(object)(model=model, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim), False
        elif backend == "gcam":
            return create_grad_cam(object)(model=model, target_layers=target_layers, postprocessor=postprocessor, retain_graph=retain_graph, dim=dim), True
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
                self._save_attention_map(attention_map_j, layer_output_dir)
                if self.evaluate:
                    score = self._comp_score(attention_map_j, batch, mask[j].squeeze())
                    #self._log_results(score, layer_name, batch_id, j, batch_size)
                    batch_scores[layer_name].append(score)
                    self.scores[layer_name].append(score)
        return batch_scores

    def _save_attention_map(self, attention_map, layer_output_dir):
        if self.save_pickle:
            self.pickle_maps.append(attention_map)
        if self.save_maps:
            save_attention_map(filename=layer_output_dir + "/attention_map_" + str(self.counter), attention_map=attention_map, heatmap=self.heatmap, dim=self.dim)
            self.counter += 1

    def _comp_score(self, attention_map, batch, mask):  # TODO: Not multiclass compatible, maybe multiclass parameter in init?
        if self.mask_key is not None:
            mask = batch[self.mask_key]
        elif mask is None:
            raise AttributeError("Either mask_key during initialization or mask during forward needs to be set")
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        else:
            mask = np.asarray(mask)
        allowed = [0, 1, 0.0, 1.0]
        if np.min(mask) in allowed and np.max(mask) in allowed:
            mask = mask.astype(int)
        else:
            raise TypeError("Mask values need to be 0/1")
        binary_attention_map, mask, weights = score_metrics.preprocessing(attention_map, mask, self.threshold)
        if self.metric[0] != "w":
            weights = None
        if self.metric == "ioa" or self.metric == "wioa":
            score = score_metrics.intersection_over_attention(binary_attention_map, mask, weights)
        elif self.metric == "iou" or self.metric == "wiou":
            score = score_metrics.intersection_over_union(binary_attention_map, mask, weights)
        else:
            score = self.metric(attention_map, mask, attention_map, weights)
        return score

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

    def __getattr__(self, method):
        def abstract_method(*args, **kwargs):
            # print("-------------------------- ABSTRACT METHOD GCAM HOOK (" + method + ") --------------------------")
            if args == () and kwargs == {}:
                return self._copy_func(getattr(self.model, method))(self)
            elif args == ():
                return self._copy_func(getattr(self.model, method))(self, **kwargs)
            elif kwargs == {}:
                return self._copy_func(getattr(self.model, method))(self, *args)
            else:
                return self._copy_func(getattr(self.model, method))(self, *args, **kwargs)

        return abstract_method

    def _copy_func(self, f):
        # print("-------------------------- COPY FUNC GCAM HOOK --------------------------")
        g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                               argdefs=f.__defaults__,
                               closure=f.__closure__)
        g = functools.update_wrapper(g, f)
        g.__kwdefaults__ = f.__kwdefaults__
        return g


# def gcam_hook(model):
#     # model_base = type(model).__bases__[0]
#     return create_gcam_hook(object)
#
# def create_gcam_hook(base):
#
#
#     return GcamHook