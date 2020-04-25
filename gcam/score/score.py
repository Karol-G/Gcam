from gcam.score import score_utils
import pandas as pd
import numpy as np

class Score():

    def __init__(self, save_path, metric="wioa", threshold='otsu', layer_ordering=None):
        self.scores = pd.DataFrame(columns=['score', 'layer', 'class_label', 'name'])
        self.save_path = save_path
        self.metric = metric
        self.threshold = threshold
        self.layer_ordering = layer_ordering

    def comp_score(self, attention_map, mask, layer=None, class_label=None, name=None):
        score = score_utils.comp_score(attention_map, mask, self.metric, self.threshold)
        self.add(score, layer, class_label, name)
        return score

    def add(self, score, layer=None, class_label=None, name=None):
        self.scores = self.scores.append({'score': score, 'layer': layer, 'class_label': class_label, 'name': name}, ignore_index=True)

    def dump(self, mean_only=False, layer=None, class_label=None):
        scores = self.scores
        if layer is not None:
            scores = scores[scores['layer']==layer]
        if class_label is not None:
            scores = scores[scores['class_label'] == class_label]
        mean_scores = self._comp_means(scores)
        with pd.ExcelWriter(self.save_path + 'scores.xlsx') as writer:
            if not mean_only:
                scores.to_excel(writer, sheet_name='Scores')
            mean_scores.to_excel(writer, sheet_name='Mean Scores')

    def _comp_means(self, scores):
        mean_scores = pd.DataFrame(columns=['mean_score', 'layer', 'class_label'])
        unique_layers = pd.unique(scores['layer'])
        if self.layer_ordering is not None:
            unique_layers = sorted(set(self.layer_ordering).intersection(unique_layers), key=lambda x: self.layer_ordering.index(x))
        for unique_layer in unique_layers:
            _scores = scores[scores['layer'] == unique_layer]
            unique_class_labels = pd.unique(_scores['class_label'])
            for unique_class_label in unique_class_labels:
                __scores = _scores[_scores['class_label'] == unique_class_label]
                mean_score = __scores['score'].to_numpy()
                mean_score = np.mean(mean_score)
                mean_scores = mean_scores.append({'mean_score': mean_score, 'layer': unique_layer, 'class_label': unique_class_label}, ignore_index=True)
        return mean_scores