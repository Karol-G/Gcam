from tests.test_model_segmentation.unet_seg_dataset import UnetSegDataset as Dataset
from tests.test_model_segmentation.unet_seg_model import UnetSegModel as Model
from gcam import gcam
import torch
import os
from os import path
import shutil
import unittest
from torch.utils.data import DataLoader


class TestSegmentation(unittest.TestCase):

    def test_forward_disabled(self):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = Dataset(device=DEVICE)
        model = Model(device=DEVICE)
        outputs = gcam.forward_disabled(model, dataset, iterations=2)
        assert outputs[0].shape == (1, 1, 384, 575)
        assert outputs[1].shape == (1, 1, 384, 575)

    def test_forward_gcam(self):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = Dataset(device=DEVICE)
        model = Model(device=DEVICE)
        batch = dataset.__getitem__(0)
        _, heatmap_gcam, heatmap_guided_gcam = gcam.forward_gcam(model, batch, layer='model.outc.conv')
        assert heatmap_gcam.shape == (1, 384, 575, 3)
        assert heatmap_guided_gcam.shape == (1, 384, 575)

    def test_evaluation_gcam(self):
        if os.path.isdir("results"):
            shutil.rmtree("results")
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = Dataset(device=DEVICE)
        model = Model(device=DEVICE)
        layer = 'auto'
        gcam.evaluate_gcam(model, dataset, result_dir="results/unet_seg/test_evaluation_gcam", layer=layer)
        assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/attention_map_0_score_100.0.png")
        assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/attention_map_1_score_100.0.png")
        assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/attention_map_2_score_100.0.png")
        assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/overlap_percentage.npy")
        assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/overlap_percentage.txt")
        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_gcam_hook(self):
        if os.path.isdir("results"):
            shutil.rmtree("results")
        DEVICE = "cpu"#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = Dataset(device=DEVICE)
        model = Model(device=DEVICE)
        layer = 'auto'
        model = gcam.gcam_hook(model)(model, is_backward_ready=True, output_dir="results/unet_seg/test_gcam_hook", layer=layer, input_key=None, mask_key=None)
        model.eval()
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch["img"])
            outputs.append(output)

        assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

if __name__ == '__main__':
    unittest.main()
