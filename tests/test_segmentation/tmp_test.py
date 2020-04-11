from tests.test_segmentation.unet_seg_dataset import UnetSegDataset as Dataset
from tests.test_segmentation.model.unet.unet_model import UNet
from gcam import gcam
import torch
import os
from os import path
import shutil
import unittest
from torch.utils.data import DataLoader
import gc


class Tmp():

    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Dataset(device=self.DEVICE)
        current_path = os.path.dirname(os.path.abspath(__file__))
        CECKPOINT_PATH = os.path.join(current_path, 'model/CHECKPOINT.pth')
        self.model = UNet(n_channels=3, n_classes=1)
        self.model.load_state_dict(torch.load(CECKPOINT_PATH, map_location=self.DEVICE))
        self.model.to(device=self.DEVICE)
        self.model.eval()

    def test_gbp_hook(self):
        model = gcam.inject(self.model, output_dir="results/unet_seg/test_gbp_hook", backend="gbp", input_key=None, mask_key=None, postprocessor="sigmoid")
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch["img"])
            #outputs.append(output)

        gc.collect()
        torch.cuda.empty_cache()

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_gcam_hook(self):
        layer = 'full'
        model = gcam.inject(self.model, output_dir="results/unet_seg/test_gcam_hook", backend="gcam", layer=layer,
                            postprocessor="sigmoid", evaluate=True, save_log=True, save_maps=True, save_pickle=True, call_dump=True)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch["img"], batch_id=i, mask=batch["gt"])
            #outputs.append(output)

        model.dump()
        gc.collect()
        torch.cuda.empty_cache()

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_ggcam_hook(self):
        layer = 'full'
        model = gcam.inject(self.model, output_dir="results/unet_seg/test_ggcam_hook", backend="ggcam", layer=layer, input_key=None, mask_key=None, postprocessor="sigmoid", save_maps=True)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch["img"])
            #outputs.append(output)

        gc.collect()
        torch.cuda.empty_cache()

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_gcampp_hook(self):
        layer = 'full'
        model = gcam.inject(self.model, output_dir="results/unet_seg/test_gcampp_hook", backend="gcampp", layer=layer, input_key=None, mask_key=None, postprocessor="sigmoid", save_maps=True)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch["img"])
            #outputs.append(output)

        gc.collect()
        torch.cuda.empty_cache()

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_gcam_hook_attribute_copy(self):
        layer = 'full'
        gcam_model = gcam.inject(self.model, output_dir="results/unet_seg/test_gcam_hook", backend="gcam", layer=layer, input_key=None, mask_key=None, postprocessor="sigmoid", save_maps=True)

        self.model.set_value(1)
        assert(self.model.get_value() == 1)
        assert (gcam_model.get_value() == -1)
        gcam_model.set_value(2)
        assert (self.model.get_value() == 1)
        assert (gcam_model.get_value() == 2)

if __name__ == '__main__':
    test = Tmp()
    #test.test_gbp_hook()
    test.test_gcam_hook()
    #test.test_ggcam_hook()
    #test.test_gcampp_hook()
    #test.test_gcam_hook_attribute_copy()