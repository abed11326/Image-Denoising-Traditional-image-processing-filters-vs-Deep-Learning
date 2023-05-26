import torch
from torch.nn import Module, Conv2d, ConvTranspose2d, ReLU, BatchNorm2d, Sequential, Flatten, Linear, MaxPool2d, Identity, Dropout, Dropout2d

def ConvBlock(inp, out, bn=False, pooling=False):
    return Sequential(
        Conv2d(inp, out, 3, 1, 1),
        BatchNorm2d(out) if bn else Identity(),
        ReLU(inplace=True),
        MaxPool2d(2, 2) if pooling else Identity()
    )

def ConvTransposedBlock(inp, out):
    return Sequential(
        ConvTranspose2d(inp, out, 2, 2),
        ReLU(inplace=True),
    )

class AE(Module):
    def __init__(self):
        super(AE, self).__init__()
        # encoder
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        # decoder
        self.upconv1 = ConvTransposedBlock(256, 128)
        self.conv5 = ConvBlock(256, 128)
        self.upconv2 = ConvTransposedBlock(128, 64)
        self.conv6 = ConvBlock(128, 64)
        self.upconv3 = ConvTransposedBlock(64, 32)
        self.conv7 = ConvBlock(64, 32)
        self.conv8 = Conv2d(32, 3, 3, padding=1)

        self.pool = MaxPool2d(2, 2)
        self.dropout = Dropout2d(p=0.25)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x = self.conv4(self.pool(x3))
        x = self.dropout(x)
        # decoder
        x = self.upconv1(x)
        x = self.conv5(torch.cat([x, x3], dim=1))
        x = self.upconv2(x)
        x = self.conv6(torch.cat([x, x2], dim=1))
        x = self.upconv3(x)
        x = self.conv7(torch.cat([x, x1], dim=1))
        x = self.conv8(x)
        return torch.sigmoid(x)
