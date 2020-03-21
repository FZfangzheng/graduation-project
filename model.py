import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ssim import msssim

NUM_BANDS = 6


def getads(array):
    amount = 0
    total = 0
    for item in array.flat:
        amount += 1
        total += item
    return total/amount


def conv(in_channels, out_channels, size, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d((size-1)/2),
        nn.Conv2d(in_channels, out_channels, size, stride=stride)
    )


# def interpolate(inputs, size=None, scale_factor=None):
#     return F.interpolate(inputs, size=size, scale_factor=scale_factor,
#                          mode='bilinear', align_corners=True)


class CompoundLoss(nn.Module):
    # def __init__(self, pretrained, alpha=0.5, normalize=True):
    #     super(CompoundLoss, self).__init__()
    #     self.pretrained = pretrained
    #     self.alpha = alpha
    #     self.normalize = normalize
    #
    # def forward(self, prediction, target):
    #     return (F.mse_loss(prediction, target) +
    #             F.mse_loss(self.pretrained(prediction), self.pretrained(target)) +
    #             self.alpha * (1.0 - msssim(prediction, target,
    #                                        normalize=self.normalize)))
    def __init__(self, alpha=0.5, normalize=True):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, prediction, target):
        return (F.mse_loss(prediction, target) +
                self.alpha * (1.0 - msssim(prediction, target,
                                           normalize=self.normalize)))


class DCNN_mapping1(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS*2, 32, 16, NUM_BANDS]
        super(DCNN_mapping1, self).__init__(
            conv(channels[0], channels[1], 9),
            nn.ReLU(True),
            conv(channels[1], channels[2], 5),
            nn.ReLU(True),
            conv(channels[2], channels[3], 5)
        )


class DCNN_mapping2(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS*2, 32, 16, NUM_BANDS]
        super(DCNN_mapping2, self).__init__(
            conv(channels[0], channels[1], 9),
            nn.ReLU(True),
            conv(channels[1], channels[2], 5),
            nn.ReLU(True),
            conv(channels[2], channels[3], 5)
        )


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.dm1 = DCNN_mapping1()
        self.dm2 = DCNN_mapping1()

    def forward(self, inputs):
        c1 = inputs[0]
        f1 = inputs[1]
        c3 = inputs[2]
        f3 = inputs[3]
        c2 = inputs[4]
        c12 = c2 - c1
        c23 = c3 - c2
        if self.training:
            c13 = c3 - c1
            f13 = f3 - f1
            print(c13.shape)
            print(f1.shape)
            pre1_f13 = self.dm1(torch.cat(c13, f1), 1)
            pre2_f13 = self.dm2(torch.cat(c13, f3), 1)
            pre_f13 = self.dm1(torch.cat(c12, f1), 1) + self.dm2(torch.cat(c23, f3), 1)
            return pre1_f13, pre2_f13, pre_f13, f13
        else:
            f12 = self.dm1(torch.cat((c12, f1), 1))
            f23 = self.dm2(torch.cat((c23, f3), 1))

            vc12 = getads(np.array(c12))
            vc23 = getads(np.array(c23))

            if vc23 - vc12 > 0.2:
                result = f1 + f12
            elif vc12 - vc23 > 0.2:
                result = f3 - f23
            else:
                canshu_alpha = (1 / vc12) / ((1 / vc12) + (1 / vc23))
                result = canshu_alpha * (f1 + f12) + (1 - canshu_alpha) * (f3 - f23)
            return result
