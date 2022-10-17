import torch
import torch.nn as nn
from .TCN import TemporalConvNet
from thop import profile
from .blocks import ReLUConvBn, ResDepSepBlock


class EEG_CNN(nn.Module):
    def __init__(self, channel):
        super(EEG_CNN, self).__init__()

        self.net = nn.Sequential(
            ReLUConvBn(channel, channel * 2, kernel_size=9, stride=4, padding=0),
            ReLUConvBn(channel * 2, channel * 2, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(channel * 2, channel * 2, kernel_size=3, stride=1, padding=0),

            ReLUConvBn(channel * 2, channel * 4, kernel_size=9, stride=4, padding=0),
            ReLUConvBn(channel * 4, channel * 4, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(channel * 4, channel * 4, kernel_size=3, stride=1, padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.out = nn.Sequential(
            nn.Linear(channel * 4, channel * 2),
            nn.ReLU(inplace=False),
            nn.Linear(channel * 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x)
        return self.out(x), x

class EEG_ResCNN(nn.Module):
    def __init__(self, channel):
        super(EEG_ResCNN, self).__init__()

        self.net = nn.Sequential(
            ResDepSepBlock(channel * 1, channel * 2, kernel_size=9, stride=4),
            ResDepSepBlock(channel * 2, channel * 2, kernel_size=3, stride=1),
            ResDepSepBlock(channel * 2, channel * 2, kernel_size=3, stride=1),
            ResDepSepBlock(channel * 2, channel * 4, kernel_size=9, stride=4),
            ResDepSepBlock(channel * 4, channel * 4, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool1d(1),
        )

        self.out = nn.Sequential(
            nn.Linear(channel * 4, channel * 1),
            nn.ReLU6(),
            nn.Linear(channel * 1, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x)
        return self.out(x), x

class EEG_TCN(nn.Module):
    def __init__(self, channel):
        super(EEG_TCN, self).__init__()

        self.net = TemporalConvNet(num_inputs=channel, num_channels=[20, 10, 5, 1])

        self.out = nn.Sequential(
            nn.Linear(600, 128),
            nn.ReLU6(inplace=False),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x)
        return self.out(x)


if __name__ == '__main__':
    from thop import profile

    model = EEG_ResCNN(channel=30)
    x = torch.randn(16, 30, 600)
    output, z = model(x)

    flops, params = profile(model, inputs=(x))
    print(flops, params)