from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class extended_skip(nn.Module):
    """
    extended_skip layer
    """

    def __init__(self, ch):
        super(extended_skip, self).__init__()

        self.gcn15 = GCN(ch, 12, 15)
        self.gcn13 = GCN(ch, 12, 13)
        self.gcn11 = GCN(ch, 12, 11)
        self.gcn9 = GCN(ch, 12, 9)
        self.gcn1 = nn.Conv2d(ch, ch, kernel_size=1)

        self.conv = conv_block(12, 12)

    def forward(self, x):
        out1 = self.gcn15(x)
        out2 = self.gcn13(x)
        out3 = self.gcn11(x)
        out4 = self.gcn9(x)

        out = out1 + out2 + out3 + out4
        out = self.conv(out)
        out5 = self.gcn1(x)
        out = torch.cat((out, out5), dim=1)

        return out


class GCN(nn.Module):
    """
    Global Convolutional Network
    """
    def __init__(self, c, out_c, k=13):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k, 1), padding=((k - 1) // 2, 0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, (k - 1) // 2))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k), padding=((k - 1) // 2, 0))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(0, (k - 1) // 2))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class ESNet(nn.Module):
    """
    ESNet Model
    """

    def __init__(self, in_ch=3, n_class=1, n=32):
        super(ESNet, self).__init__()

        filters = [n, n * 2, n * 4, n * 8]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.Skip1 = extended_skip(filters[0])
        self.Skip2 = extended_skip(filters[1])
        self.Skip3 = extended_skip(filters[2])

        self.bottom = GCN(filters[3], filters[3])

        self.Deconv4 = up_conv(filters[3], n)
        self.double4 = conv_block(n, n)

        self.Deconv3 = up_conv(filters[2] + n + 12, n)
        self.double3 = conv_block(n, n)

        self.Deconv2 = up_conv(filters[1] + n + 12, n)
        self.double2 = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(n),
            nn.ReLU(inplace=True)
        )

        self.Deconv1 = conv_block(filters[0] + n + 12, n)
        self.relu = nn.ReLU(inplace=True)

        self.Conv_last = nn.Conv2d(filters[0], n_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # downsampling
        conv1 = self.Conv1(x)
        x = self.pool(conv1)
        conv2 = self.Conv2(x)
        x = self.pool(conv2)
        conv3 = self.Conv3(x)
        x = self.pool(conv3)
        conv4 = self.Conv4(x)

        # skip connections
        skip1 = self.Skip1(conv1)
        skip2 = self.Skip2(conv2)
        skip3 = self.Skip3(conv3)

        # bottom layers
        x = self.relu(self.bottom(conv4))
        x = self.relu(self.bottom(x))

        # upsamling
        deconv4 = self.Deconv4(x)
        x = self.double4(deconv4)
        x = torch.cat((x, skip3), dim=1)

        deconv3 = self.Deconv3(x)
        x = self.double3(deconv3)
        x = torch.cat((x, skip2), dim=1)

        deconv2 = self.Deconv2(x)
        x = self.double2(deconv2)
        x = torch.cat([x, skip1], dim=1)
        x = self.Deconv1(x)

        # final layer
        x = self.Conv_last(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
