import torch
from collections import Sequence
from tqdm import tqdm
from torch.nn import functional as F

def create_occlusion_sensitivity(base):  # TODO: WORK IN PROGRESS
    class OcclusionSensitivity(base):

        def __init__(self, model, mean=0, patch=35, stride=1, n_batches=128):
            self.model = model
            self.mean = mean
            self.patch = patch
            self.stride = stride
            self.n_batches = n_batches

        def forward(self, data, data_shape):
            with torch.no_grad():
                patch_H, patch_W = self.patch if isinstance(self.patch, Sequence) else (self.patch, self.patch)
                pad_H, pad_W = patch_H // 2, patch_W // 2

                # Padded image
                images = F.pad(data, (pad_W, pad_W, pad_H, pad_H), value=self.mean)
                B, _, H, W = images.shape
                new_H = (H - patch_H) // self.stride + 1
                new_W = (W - patch_W) // self.stride + 1

                # Prepare sampling grids
                anchors = []
                grid_h = 0
                while grid_h <= H - patch_H:
                    grid_w = 0
                    while grid_w <= W - patch_W:
                        grid_w += self.stride
                        anchors.append((grid_h, grid_w))
                    grid_h += self.stride

                # Baseline score without occlusion
                baseline = self.model(images).detach().gather(1, ids)

                # Compute per-pixel logits
                scoremaps = []
                for i in tqdm(range(0, len(anchors), self.n_batches), leave=False):
                    batch_images = []
                    batch_ids = []
                    for grid_h, grid_w in anchors[i: i + self.n_batches]:
                        images_ = images.clone()
                        images_[..., grid_h: grid_h + patch_H, grid_w: grid_w + patch_W] = self.mean
                        batch_images.append(images_)
                        batch_ids.append(ids)
                    batch_images = torch.cat(batch_images, dim=0)
                    batch_ids = torch.cat(batch_ids, dim=0)
                    scores = self.model(batch_images).detach().gather(1, batch_ids)
                    scoremaps += list(torch.split(scores, B))

                diffmaps = torch.cat(scoremaps, dim=1) - baseline
                diffmaps = diffmaps.view(B, new_H, new_W)

                return diffmaps

        def backward(self, output=None, label=None):
            return

        def generate(self):
            pass

    return OcclusionSensitivity

def occlusion_sensitivity(model, images, ids, mean=None, patch=35, stride=1, n_batches=128):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17
    Originally proposed in:
    "Visualizing and Understanding Convolutional Networks"
    https://arxiv.org/abs/1311.2901
    """

    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(images).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        batch_images = []
        batch_ids = []
        for grid_h, grid_w in anchors[i : i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_ids.append(ids)
        batch_images = torch.cat(batch_images, dim=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        scores = model(batch_images).detach().gather(1, batch_ids)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps