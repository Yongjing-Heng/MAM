import torch
import torch.nn as nn

class MixedAttentionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(MixedAttentionBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels//4, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels//4, output_channels//4, 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels//4, output_channels, 1, 1, bias = False)
        self.ca = ChannelAttention(output_channels)
        self.sa = SpatialAttention()        
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)


    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.ca(out) * out
        out = self.sa(out) * out

        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        min_out = self.fc2(self.relu1(self.fc1(-self.max_pool(-x))))
        out = self.sigmoid(avg_out + max_out) + self.tanh(min_out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1,
                                    output_padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
        self.maxpooling = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.sigmoid(self.conv1(x1))
        
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        min_out = self.relu(self.deconv(min_out))  # >>> 1x2hx2w
        min_out = self.conv2(min_out)
        min_out = -self.maxpooling(-min_out)  # >>> 1xhxw
        x2 = self.tanh(min_out)

        return x1 + x2
