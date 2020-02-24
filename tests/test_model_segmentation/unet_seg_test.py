from tests.test_model_segmentation.unet_seg_dataset import UnetSegDataset as Dataset
from tests.test_model_segmentation.unet_seg_model import UnetSegModel as Model
from gcam import gcam
import torch
import os
from os import path
import shutil
import unittest


class TestSegmentation(unittest.TestCase):

    def test_segmentation(self):
        if os.path.isdir("results"):
            shutil.rmtree("results")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = Dataset(device=DEVICE)
        model = Model(device=DEVICE)
        gcam.evaluate_gcam(model, dataset, result_dir="results/unet_seg", layer='model.outc.conv')
        assert path.exists("results/unet_seg/model.outc.conv/attention_map_0_score_100.0.png")
        assert path.exists("results/unet_seg/model.outc.conv/attention_map_1_score_100.0.png")
        assert path.exists("results/unet_seg/model.outc.conv/attention_map_2_score_100.0.png")
        assert path.exists("results/unet_seg/model.outc.conv/overlap_percentage.npy")
        assert path.exists("results/unet_seg/model.outc.conv/overlap_percentage.txt")
        if os.path.isdir("results"):
            shutil.rmtree("results")

if __name__ == '__main__':
    unittest.main()
