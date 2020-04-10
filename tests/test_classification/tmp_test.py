from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from gcam import gcam
import torch
import cv2
from torch.utils.data import DataLoader
import gc


class Tmp():

    def __init__(self):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = ImageFolder("data", loader=self.load_image)
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

    def test_gbp_hook(self):
        model = gcam.inject(self.model, output_dir="results/resnet152/test_gbp_hook", backend="gbp", input_key=None, mask_key=None, postprocessor="softmax")
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch[0])
            #outputs.append(output)

        gc.collect()
        torch.cuda.empty_cache()

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_gcam_hook(self):
        layer = 'layer4'
        model = gcam.inject(self.model, output_dir="results/resnet152/test_gcam_hook", backend="gcam", layer=layer, input_key=None, mask_key=None, postprocessor="softmax")
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch[0], label="best")
            #outputs.append(output)

        gc.collect()
        torch.cuda.empty_cache()

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_ggcam_hook(self):
        layer = 'layer4'
        model = gcam.inject(self.model, output_dir="results/resnet152/test_ggcam_hook", backend="ggcam", layer=layer, input_key=None, mask_key=None, postprocessor="softmax")
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch[0], label="best")
            #outputs.append(output)

        gc.collect()
        torch.cuda.empty_cache()

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

    def test_gcampp_hook(self):
        layer = 'layer4'
        model = gcam.inject(self.model, output_dir="results/resnet152/test_gcampp_hook", backend="gcampp", layer=layer, input_key=None, mask_key=None, postprocessor="softmax")
        model.eval()
        data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        # TODO: Memory leak finden (Oder nur beim testen?)
        #outputs = []
        for i, batch in enumerate(data_loader):
            output = model(batch[0], label="best")
            #outputs.append(output)

        gc.collect()
        torch.cuda.empty_cache()

        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_0.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_1.png")
        # assert path.exists("results/unet_seg/test_gcam_hook/" + layer + "/attention_map_2.png")

        # if os.path.isdir("results"):
        #     shutil.rmtree("results")

if __name__ == '__main__':
    test = Tmp()
    test.test_gbp_hook()
    test.test_gcam_hook()
    test.test_ggcam_hook()
    test.test_gcampp_hook()
