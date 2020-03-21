import torch.nn as nn


NUM_BANDS = 6


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


def deconv3x3(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 3,
                              stride=2, padding=1, output_padding=1)


# MODIS低分辨率网络
class HTLSNet(nn.Module):
    def __init__(self):
        super(HTLSNet, self).__init__()
        channels = [32, 64, 128]
        self.conv1 = conv3x3(NUM_BANDS, channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels[0], channels[1])
        self.deconvs = nn.ModuleList([deconv3x3(channels[1], channels[1]) for _ in range(3)])
        self.conv3 = conv3x3(channels[1], channels[2])

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        for layer in self.deconvs:
            out = layer(out)
            out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        return out


# Landsat高分辨率网络
class LTHSNet(nn.Module):
    def __init__(self):
        super(LTHSNet, self).__init__()
        channels = [32, 64, 128]
        self.conv1 = conv3x3(NUM_BANDS, channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels[0], channels[1])
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.conv3 = conv3x3(channels[1], channels[1])
        self.conv4 = conv3x3(channels[1], channels[2])

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pooling(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        return out


# Reconstruction重建图像网络
class ReconstructNet(nn.Module):
    def __init__(self):
        super(ReconstructNet, self).__init__()
        channels = [32, 64, 128]
        self.deconv = deconv3x3(channels[-1], channels[-2])
        self.relu = nn.ReLU(inplace=True)
        self.dense1 = nn.Linear(channels[-2], channels[-3])
        self.dense2 = nn.Linear(channels[-3], NUM_BANDS)

    def forward(self, x):
        out = x
        out = self.deconv(out)
        out = self.relu(out)
        out = out.permute(0, 2, 3, 1)
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dense2(out)
        out = out.permute(0, 3, 1, 2)
        return out


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.coarse_net = HTLSNet()
        self.fine_net = LTHSNet()
        self.reconstruct_net = ReconstructNet()

    def forward(self, inputs):
        modis1 = inputs[0]
        landsat1 = inputs[1]
        modis = inputs[2]
        print(inputs[0].shape)
        print(inputs[1].shape)
        print(inputs[2].shape)
        htls = self.coarse_net(modis)

        htls1 = self.coarse_net(modis1)
        lths1 = self.fine_net(landsat1)
        print(htls.shape)
        print(lths1.shape)
        print(htls1.shape)
        result = htls + lths1 - htls1

        result = self.reconstruct_net(result)
        return result
