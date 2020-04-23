from gcam import gcam_inject
from gcam import gcam_utils
from gcam import score_utils


def inject(*args, **kwargs):
    gcam_inject.inject(*args, **kwargs)


def get_layers(model, reverse=False):
    return gcam_utils.get_layers(model, reverse)


def compute_score(attention_map, mask, metric="wioa", threshold=0.3):
    return score_utils.comp_score(attention_map, mask, metric, threshold)

def save(attention_map, filename, heatmap):
    gcam_utils.save_attention_map(filename, attention_map, heatmap)