from tests.test_model_segmentation.unet_seg_dataset import UnetSegDataset as Dataset
from tests.test_model_segmentation.unet_seg_model import UnetSegModel as Model
from tests.test_model_segmentation.model.unet.unet_model import UNet
from gcam import gcam
import torch
import os
from os import path
import shutil
import unittest
from torch.utils.data import DataLoader


class TestSegmentation(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSegmentation, self).__init__(*args, **kwargs)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Dataset(device=self.DEVICE)
        # model = Model(device=DEVICE) #TODO: Sigmoid missing
        current_path = os.path.dirname(os.path.abspath(__file__))
        CECKPOINT_PATH = os.path.join(current_path, 'model/CHECKPOINT.pth')
        self.model = UNet(n_channels=3, n_classes=1)
        self.model.load_state_dict(torch.load(CECKPOINT_PATH, map_location=self.DEVICE))
        self.model.to(device=self.DEVICE)
        self.model.eval()

    # def test_forward_disabled(self):
    #     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     dataset = Dataset(device=DEVICE)
    #     model = Model(device=DEVICE)
    #     outputs = gcam.forward_disabled(model, dataset, iterations=2)
    #     assert outputs[0].shape == (1, 1, 384, 575)
    #     assert outputs[1].shape == (1, 1, 384, 575)

    # def test_forward_gcam(self):
    #     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     dataset = Dataset(device=DEVICE)
    #     model = Model(device=DEVICE)
    #     batch = dataset.__getitem__(0)
    #     _, heatmap_gcam, heatmap_guided_gcam = gcam.forward_gcam(model, batch, layer='model.outc.conv')
    #     assert heatmap_gcam.shape == (1, 384, 575, 3)
    #     assert heatmap_guided_gcam.shape == (1, 384, 575)

    # def test_evaluation_gcam(self):
    #     if os.path.isdir("results"):
    #         shutil.rmtree("results")
    #     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     dataset = Dataset(device=DEVICE)
    #     model = Model(device=DEVICE)
    #     layer = 'full' #'model.outc.conv'
    #     gcam.extract(model, dataset, output_dir="results/unet_seg/test_evaluation_gcam", layer=layer)
    #     assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/attention_map_0_score_100.0.png")
    #     assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/attention_map_1_score_100.0.png")
    #     assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/attention_map_2_score_100.0.png")
    #     # assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/overlap_percentage.npy")
    #     # assert path.exists("results/unet_seg/test_evaluation_gcam/" + layer + "/overlap_percentage.txt")
    #     # if os.path.isdir("results"):
    #     #     shutil.rmtree("results")

    def test_gcam_hook(self):
        layer = 'full'
        model = gcam.inject(self.model, is_backward_ready=True, output_dir="results/unet_seg/test_gcam_hook", layer=layer, input_key=None, mask_key=None, postprocessor="sigmoid")
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch["img"])
            #outputs.append(output)

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_gcam_hook_attribute_copy(self):
        layer = 'full'
        gcam_model = gcam.inject(self.model, is_backward_ready=True, output_dir="results/unet_seg/test_gcam_hook", layer=layer, input_key=None, mask_key=None, postprocessor="sigmoid")

        self.model.set_value(1)
        assert(self.model.get_value() == 1)
        assert (gcam_model.get_value() == -1)
        gcam_model.set_value(2)
        assert (self.model.get_value() == 1)
        assert (gcam_model.get_value() == 2)

if __name__ == '__main__':
    unittest.main()


    # def test_gcam_hook(self):
    #     if os.path.isdir("results"):
    #         shutil.rmtree("results")
    #     DEVICE = "cpu"#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     dataset = Dataset(device=DEVICE)
    #     model = Model(device=DEVICE)
    #     class MyCNN():
    #         pass
    #
    #
    #
    #     model = MyCNN()
    #     # Import gcam
    #     from gcam import gcam
    #     # Hook forward method of model with gcam
    #     model = gcam.gcam_hook(model, output_dir="results", layer='auto')
    #     # Continue to do what your doing...
    #     # In this case inference on some new data
    #     model.eval()
    #     data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #     for i, batch in enumerate(data_loader):
    #         output = model(batch)
    #         # more code...
    #
    #
    #
    #
    #         outputs.append(output)

    # def test_gcam_hook(self):
    #     if os.path.isdir("results"):
    #         shutil.rmtree("results")
    #     DEVICE = "cpu"#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     dataset = Dataset(device=DEVICE)
    #     model = Model(device=DEVICE)
    #     class MyCNN():
    #         pass
    #     class MyDataset():
    #         pass
    #
    #
    #     class WrapMyCNN():
    #         # Wraps MyCNN and keeps some requirements in mind
    #         #...
    #         pass
    #
    #     model = WrapMyCNN()
    #     dataset = MyDataset()
    #     # Import gcam
    #     from gcam import gcam
    #     # Evaluate dataset
    #     gcam.evaluate(model, dataset, output_dir="results", layer='auto', input_key="img", mask_key="gt")
    #
    #
    #
    #
    #         outputs.append(output)

