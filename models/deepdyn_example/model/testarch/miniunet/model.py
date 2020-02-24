import torch
import torch.nn.functional as F
from torch import nn

from utils.weights_utils import initialize_weights


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, p=0):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channel, kernel_size=3, padding=p),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, out_channels, kernel_size=3, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class MiniUNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(MiniUNet, self).__init__()

        reduce_by = 4

        self.A3_ = _DoubleConvolution(num_channels, int(256 / reduce_by), int(256 / reduce_by))
        self.A4_ = _DoubleConvolution(int(256 / reduce_by), int(512 / reduce_by), int(512 / reduce_by))

        self.A_mid = _DoubleConvolution(int(512 / reduce_by), int(1024 / reduce_by), int(1024 / reduce_by))

        self.A4_up = nn.ConvTranspose2d(int(1024 / reduce_by), int(512 / reduce_by), kernel_size=2, stride=2)
        self._A4 = _DoubleConvolution(int(1024 / reduce_by), int(512 / reduce_by), int(512 / reduce_by))

        self.A3_up = nn.ConvTranspose2d(int(512 / reduce_by), int(256 / reduce_by), kernel_size=2, stride=2)
        self._A3 = _DoubleConvolution(int(512 / reduce_by), int(256 / reduce_by), int(256 / reduce_by))

        self.final = nn.Conv2d(int(256 / reduce_by), num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        a3_ = self.A3_(x)
        a3_dwn = F.max_pool2d(a3_, kernel_size=2, stride=2)

        a4_ = self.A4_(a3_dwn)
        a4_dwn = F.max_pool2d(a4_, kernel_size=2, stride=2)

        a_mid = self.A_mid(a4_dwn)

        a4_up = self.A4_up(a_mid)
        _a4 = self._A4(MiniUNet.match_and_concat(a4_, a4_up))

        a3_up = self.A3_up(_a4)
        _a3 = self._A3(MiniUNet.match_and_concat(a3_, a3_up))

        final = self.final(_a3)
        return final

    @staticmethod
    def match_and_concat(bypass, upsampled, crop=True):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)


m = MiniUNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
