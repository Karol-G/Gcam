from gcam import gcam_inject
from gcam import gcam_utils
from gcam.score import score_utils, score


def inject(*args, **kwargs):
    return gcam_inject.inject(*args, **kwargs)


def get_layers(model, reverse=False):
    return gcam_utils.get_layers(model, reverse)


def compute_score(attention_map, mask, metric="wioa", threshold='otsu'):
    return score_utils.comp_score(attention_map, mask, metric, threshold)


def Score(save_path, metric="wioa", threshold='otsu', layer_ordering=None):
    return score.Score(save_path, metric, threshold, layer_ordering)


def save(attention_map, filename, heatmap):
    gcam_utils.save_attention_map(filename, attention_map, heatmap)