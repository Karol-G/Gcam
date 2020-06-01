from tests.test_segmentation.unet_seg_dataset import UnetSegDataset as Dataset
from tests.test_segmentation.model.unet.unet_model import UNet
from gcam import gcam
import torch
import os
import unittest
from torch.utils.data import DataLoader
import gc
import shutil


class TestSegmentation(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSegmentation, self).__init__(*args, **kwargs)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Dataset(device=self.DEVICE)
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        ceckpoint_path = os.path.join(self.current_path, 'model/CHECKPOINT.pth0')
        self.model = UNet(n_channels=3, n_classes=1)
        self.model.load_state_dict(torch.load(ceckpoint_path, map_location=self.DEVICE))
        self.model.to(device=self.DEVICE)
        self.model.eval()

    def test_gbp(self):
        model = gcam.inject(self.model, output_dir=os.path.join(self.current_path, 'results/unet_seg/gbp'), backend='gbp',
                            evaluate=True, save_scores=False, save_maps=False, save_pickle=False, metric="wioa", label=lambda x: 0.5 < x, channels=1)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        for i, batch in enumerate(data_loader):
            _ = model(batch["img"], mask=batch["gt"])

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if os.path.isdir(os.path.join(self.current_path, 'results/unet_seg')):
            shutil.rmtree(os.path.join(self.current_path, 'results/unet_seg'))

    def test_gcam(self):
        layer = 'full'
        metric = 'wioa'
        model = gcam.inject(self.model, output_dir=os.path.join(self.current_path, 'results/unet_seg/gcam'), backend='gcam', layer=layer,
                            evaluate=True, save_scores=False, save_maps=False, save_pickle=False, metric=metric, label=lambda x: 0.5 < x, channels=1)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        model.test_run(next(iter(data_loader))["img"])

        for i, batch in enumerate(data_loader):
            _ = model(batch["img"], mask=batch["gt"])

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if os.path.isdir(os.path.join(self.current_path, 'results/unet_seg')):
            shutil.rmtree(os.path.join(self.current_path, 'results/unet_seg'))

    def test_ggcam(self):
        layer = 'full'
        metric = 'wioa'
        model = gcam.inject(self.model, output_dir=os.path.join(self.current_path, 'results/unet_seg/ggcam'), backend='ggcam', layer=layer,
                            evaluate=True, save_scores=False, save_maps=False, save_pickle=False, metric=metric, label=lambda x: 0.5 < x, channels=1)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        model.test_run(next(iter(data_loader))["img"])

        for i, batch in enumerate(data_loader):
            _ = model(batch["img"], mask=batch["gt"])

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if os.path.isdir(os.path.join(self.current_path, 'results/unet_seg')):
            shutil.rmtree(os.path.join(self.current_path, 'results/unet_seg'))

    def test_gcampp(self):
        layer = 'full'
        metric = 'wioa'
        model = gcam.inject(self.model, output_dir=os.path.join(self.current_path, 'results/unet_seg/gcampp'), backend='gcampp', layer=layer,
                            evaluate=True, save_scores=False, save_maps=False, save_pickle=False, metric=metric, label=lambda x: 0.5 < x, channels=1)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        model.test_run(next(iter(data_loader))["img"])

        for i, batch in enumerate(data_loader):
            _ = model(batch["img"], mask=batch["gt"])

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if os.path.isdir(os.path.join(self.current_path, 'results/unet_seg')):
            shutil.rmtree(os.path.join(self.current_path, 'results/unet_seg'))


if __name__ == '__main__':
    unittest.main()
