from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from gcam import gcam
import torch
import cv2
from torch.utils.data import DataLoader
import gc
import shutil
import os
import unittest

CLEAR = True


class TestClassification(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestClassification, self).__init__(*args, **kwargs)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset = ImageFolder(os.path.join(self.current_path, 'data'), loader=self.load_image)
        self.model = models.resnet152(pretrained=True)
        self.model.to(device=self.DEVICE)
        self.model.eval()

    def load_image(self, image_path):
        raw_image = cv2.imread(image_path)
        raw_image = cv2.resize(raw_image, (224,) * 2)
        image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(raw_image[..., ::-1].copy())
        image = image.to(self.DEVICE)
        return image

    def test_gbp(self):
        model = gcam.inject(self.model, output_dir=os.path.join(self.current_path, 'results/resnet152/test_gbp'), backend='gbp',
                    evaluate=False, save_scores=False, save_maps=True, save_pickle=False, channels=1)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        for i, batch in enumerate(data_loader):
            _ = model(batch[0])

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if CLEAR and os.path.isdir(os.path.join(self.current_path, 'results/resnet152')):
            shutil.rmtree(os.path.join(self.current_path, 'results/resnet152'))

    def test_gcam(self):
        layer = 'layer4'
        model = gcam.inject(self.model, output_dir=os.path.join(self.current_path, 'results/resnet152/test_gcam'), backend='gcam', layer=layer,
                    evaluate=False, save_scores=False, save_maps=True, save_pickle=False, channels=1)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        model.test_run(next(iter(data_loader))[0])

        for i, batch in enumerate(data_loader):
            _ = model(batch[0])

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if CLEAR and os.path.isdir(os.path.join(self.current_path, 'results/resnet152')):
            shutil.rmtree(os.path.join(self.current_path, 'results/resnet152'))

    def test_ggcam(self):
        layer = 'layer4'
        model = gcam.inject(self.model, output_dir=os.path.join(self.current_path, 'results/resnet152/test_ggcam'), backend='ggcam', layer=layer,
                    evaluate=False, save_scores=False, save_maps=True, save_pickle=False, channels=1)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        model.test_run(next(iter(data_loader))[0])

        for i, batch in enumerate(data_loader):
            _ = model(batch[0])

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if CLEAR and os.path.isdir(os.path.join(self.current_path, 'results/resnet152')):
            shutil.rmtree(os.path.join(self.current_path, 'results/resnet152'))

    def test_gcampp(self):
        layer = 'layer4'
        model = gcam.inject(self.model, output_dir=os.path.join(self.current_path, 'results/resnet152/test_gcampp'), backend='gcampp', layer=layer,
                    evaluate=False, save_scores=False, save_maps=True, save_pickle=False, channels=1)
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        model.test_run(next(iter(data_loader))[0])

        for i, batch in enumerate(data_loader):
            _ = model(batch[0])

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if CLEAR and os.path.isdir(os.path.join(self.current_path, 'results/resnet152')):
            shutil.rmtree(os.path.join(self.current_path, 'results/resnet152'))

if __name__ == '__main__':
    unittest.main()
