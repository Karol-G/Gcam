from gcam import gcam_inject
from gcam import gcam_utils
from gcam.evaluation import evaluation_utils, evaluator
from functools import wraps


@wraps(gcam_inject.inject)
def inject(*args, **kwargs):
    return gcam_inject.inject(*args, **kwargs)


@wraps(gcam_utils.get_layers)
def get_layers(model, reverse=False):
    return gcam_utils.get_layers(model, reverse)


@wraps(evaluation_utils.comp_score)
def compute_score(attention_map, mask, metric="wioa", threshold='otsu'):
    return evaluation_utils.comp_score(attention_map, mask, metric, threshold)


@wraps(evaluator.Evaluator)
def Evaluator(save_path, metric="wioa", threshold='otsu', layer_ordering=None):
    return evaluator.Evaluator(save_path, metric, threshold, layer_ordering)


@wraps(gcam_utils.save_attention_map)
def save(attention_map, filename, heatmap):
    gcam_utils.save_attention_map(filename, attention_map, heatmap)