import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ssim import msssim

NUM_BANDS = 6


def getads(array):
    total = 0
    for item in array.flat:
        total += item
    return total


def conv(in_channels, out_channels, size, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(int((size-1)/2)),
        nn.Conv2d(in_channels, out_channels, size, stride=stride)
    )


def interpolate(inputs, size=None, scale_factor=None):
    return F.interpolate(inputs, size=size, scale_factor=scale_factor,
                         mode='bilinear', align_corners=True)


class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.5, normalize=True):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, prediction, target):
        return (F.mse_loss(prediction, target) +
                self.alpha * (1.0 - msssim(prediction, target,
                                           normalize=self.normalize)))


class SR(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 64, 32, NUM_BANDS]
        super(SR, self).__init__(
            conv(channels[0], channels[1], 9),
            nn.ReLU(True),
            conv(channels[1], channels[2], 5),
            nn.ReLU(True),
            conv(channels[2], channels[3], 5),
        )


class NML(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 64, 32, NUM_BANDS]
        super(NML, self).__init__(
            conv(channels[0], channels[1], 9),
            nn.ReLU(True),
            conv(channels[1], channels[2], 5),
            nn.ReLU(True),
            conv(channels[2], channels[3], 5),
        )


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.sr = SR()
        self.nml = NML()

    def forward(self, inputs):
        # inputs[1]是参考Landsat，inputs[0]和inputs[-1](即数组最后一个)是参考MODIS和目标时间的MODIS
        # 首先因为数据集给出的图像尺度相同，所以做下采样处理，下采样降低十倍分辨率
        if self.training:
            modis1 = interpolate(inputs[0], scale_factor=0.1)
            lsr_landsat1 = interpolate(inputs[1], scale_factor=0.1)
            pre_lsr_landsat1 = torch.add(modis1, self.nml(modis1))
            new_pre_lsr_landsat1 = interpolate(pre_lsr_landsat1, scale_factor=10)
            pre_landsat1 = torch.add(new_pre_lsr_landsat1, self.sr(new_pre_lsr_landsat1))
            return pre_lsr_landsat1, lsr_landsat1, pre_landsat1
        else:
            modis1 = interpolate(inputs[0], scale_factor=0.1)
            lsr_landsat1 = interpolate(inputs[1], scale_factor=0.1)
            modis3 = interpolate(inputs[2], scale_factor=0.1)
            lsr_landsat3 = interpolate(inputs[3], scale_factor=0.1)
            modis2 = interpolate(inputs[4], scale_factor=0.1)

            pre_lsr_landsat1 = modis1 + self.nml(modis1)
            pre_lsr_landsat3 = modis3 + self.nml(modis3)
            pre_lsr_landsat2 = modis2 + self.nml(modis2)

            l21 = lsr_landsat1.mul(torch.div(pre_lsr_landsat2, pre_lsr_landsat1))
            l23 = lsr_landsat3.mul(torch.div(pre_lsr_landsat2, pre_lsr_landsat3))

            p1 = 1 / getads(np.array(torch.sub(pre_lsr_landsat2, pre_lsr_landsat1)))
            p3 = 1 / getads(np.array(torch.sub(pre_lsr_landsat2, pre_lsr_landsat3)))
            w1 = p1 / (p1 + p3)
            w3 = p3 / (p1 + p3)

            new_pre_lsr_landsat2 = w1 * l21 + w3 * l23
            pre_landsat2 = self.sr(new_pre_lsr_landsat2)
            pre_landsat1 = self.sr(lsr_landsat1)
            pre_landsat3 = self.sr(lsr_landsat3)

            l21 = inputs[1].mul(torch.div(pre_landsat2, pre_landsat1))
            l23 = inputs[3].mul(torch.div(pre_landsat2, pre_landsat3))

            p1 = 1 / getads(np.array(torch.sub(pre_landsat2, pre_landsat1)))
            p3 = 1 / getads(np.array(torch.sub(pre_landsat2, pre_landsat3)))
            w1 = p1 / (p1 + p3)
            w3 = p3 / (p1 + p3)

            new_pre_landsat2 = w1 * l21 + w3 * l23

            return new_pre_landsat2
