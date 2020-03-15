import torch
import torch.nn as nn
import torch.nn.functional as F

from ssim import msssim

NUM_BANDS = 6


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )

# def interpolate(inputs, size=None, scale_factor=None):
#     return F.interpolate(inputs, size=size, scale_factor=scale_factor,
#                          mode='bilinear', align_corners=True)


class ResidualBlock(nn.Module):
    """实现一个残差块"""
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):

        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),  # 这个卷积操作是不会改变w h的
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        out += residual
        return F.relu(out)


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


class FEncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128, 256]
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True),
            conv3x3(channels[3], channels[4]),
            nn.ReLU(True)
        )


class REncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS * 3, 32, 64, 128, 256]
        super(REncoder, self).__init__(
            ResidualBlock(channels[0], channels[1]),
            ResidualBlock(channels[1], channels[2]),
            ResidualBlock(channels[2], channels[3]),
            ResidualBlock(channels[3], channels[4]),
            # conv3x3(channels[0], channels[1]),
            # nn.ReLU(True),
            # conv3x3(channels[1], channels[2]),
            # nn.ReLU(True),
            # conv3x3(channels[2], channels[3]),
            # nn.ReLU(True),
            # conv3x3(channels[3], channels[4]),
            # nn.ReLU(True)
        )


class Decoder(nn.Sequential):
    def __init__(self):
        channels = [256, 128, 64, 32, NUM_BANDS]
        super(Decoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True),
            nn.Conv2d(channels[3], channels[4], 1)
        )

class Truing(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS*2, 32, 64, 32, NUM_BANDS]
        super(Truing, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True),
            nn.Conv2d(channels[3], channels[4], 1)
        )

class Pretrained(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(Pretrained, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2], 2),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3], 2),
            nn.ReLU(True)
        )


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.encoder = FEncoder()
        self.residual = REncoder()
        self.decoder = Decoder()
        self.truing = Truing()

    def forward(self, inputs):
        # 原本是300上采样到4800，但是这里不用，给出的MODIS和Landsat一样
        # inputs[1]是参考Landsat，inputs[0]和inputs[-1](即数组最后一个)是参考MODIS和目标时间的MODIS
        # inputs[0] = interpolate(inputs[0], scale_factor=16)
        # inputs[-1] = interpolate(inputs[-1], scale_factor=16)
        # c_diff1 = torch.sub(inputs[-1], inputs[0])
        # prev_diff = self.residual(torch.cat((inputs[0], inputs[1], inputs[-1], c_diff1), 1))
        prev_diff = self.residual(torch.cat((inputs[0], inputs[1], inputs[-1]), 1))
        # len==5则表示有两对参考
        if len(inputs) == 5:
            # c_diff2 = torch.sub(inputs[2], inputs[-1])
            # next_diff = self.residual(torch.cat((inputs[2], inputs[3], inputs[-1], c_diff2), 1))
            next_diff = self.residual(torch.cat((inputs[2], inputs[3], inputs[-1]), 1))
            if self.training:
                prev_fusion = self.encoder(inputs[1]) + prev_diff
                next_fusion = self.encoder(inputs[3]) + next_diff
                return self.decoder(prev_fusion), self.decoder(next_fusion)
            else:
                # zero = inputs[0].new_tensor(0.0)
                one = inputs[0].new_tensor(1.0)
                epsilon = inputs[0].new_tensor(1e-8)
                # threshold = inputs[0].new_tensor(0.2)
                prev_dist = torch.abs(prev_diff) + epsilon
                next_dist = torch.abs(next_diff) + epsilon
                prev_mask = one.div(prev_dist).div(one.div(prev_dist) + one.div(next_dist))
                # prev_mask[prev_dist - next_dist > threshold] = zero
                # prev_mask[next_dist - prev_dist > threshold] = one
                prev_mask = prev_mask.clamp_(0.0, 1.0)
                next_mask = one - prev_mask
                result = (prev_mask * (self.encoder(inputs[1]) + prev_diff) +
                          next_mask * (self.encoder(inputs[3]) + next_diff))
                # prev_fusion = self.encoder(inputs[1]) + prev_diff
                # next_fusion = self.encoder(inputs[3]) + next_diff
                # prev_fusion[prev_dist >= next_dist] = zero
                # next_fusion[prev_dist < next_dist] = zero
                # result = prev_fusion + next_fusion
                result = self.decoder(result)
                # f_diff = torch.sub(inputs[3], inputs[1])
                # result = self.truing(torch.cat((result, f_diff), 1))
                return result
        else:
            return self.decoder(self.encoder(inputs[1]) + prev_diff)
