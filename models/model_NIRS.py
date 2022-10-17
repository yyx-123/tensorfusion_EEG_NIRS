import torch
import torch.nn as nn
from .TCN import TemporalConvNet
from .blocks import ReLUConvBn, ResDepSepBlock




class NIRS_CNN(nn.Module):
    def __init__(self, channel):
        super(NIRS_CNN, self).__init__()

        self.net = nn.Sequential(
            ReLUConvBn(channel, channel * 2, kernel_size=5, stride=2, padding=0),
            ReLUConvBn(channel * 2, channel * 2, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(channel * 2, channel * 2, kernel_size=3, stride=1, padding=0),

            ReLUConvBn(channel * 2, channel * 4, kernel_size=5, stride=1, padding=0),
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


class NIRS_ResCNN(nn.Module):
    def __init__(self, channel):
        super(NIRS_ResCNN, self).__init__()

        self.net = nn.Sequential(
            ResDepSepBlock(channel * 1, channel * 2, kernel_size=9, stride=4),
            ResDepSepBlock(channel * 2, channel * 2, kernel_size=3, stride=1),
            ResDepSepBlock(channel * 2, channel * 2, kernel_size=3, stride=1),
            ResDepSepBlock(channel * 2, channel * 4, kernel_size=9, stride=4),
            ResDepSepBlock(channel * 4, channel * 4, kernel_size=3, stride=1),
            ResDepSepBlock(channel * 4, channel * 4, kernel_size=3, stride=1),

            nn.AdaptiveAvgPool1d(1),
        )
        self.out = nn.Sequential(
            nn.Linear(channel * 4, channel * 2),
            nn.ReLU6(inplace=False),
            nn.Linear(channel * 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x)
        return self.out(x), x

class NIRS_TCN(nn.Module):
    def __init__(self, channel):
        super(NIRS_TCN, self).__init__()

        self.net = TemporalConvNet(num_inputs=channel, num_channels=[18, 9, 5, 1])

        self.out = nn.Sequential(
            nn.Linear(50, 20),
            nn.ReLU6(inplace=False),
            nn.Linear(20, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x)
        return self.out(x)

if __name__ == '__main__':
    from thop import profile

    model = NIRS_TCN(channel=36)
    x = torch.randn(16, 36, 50)
    out = model(x)
    flops, params = profile(model, inputs=(x))
    print(flops, params)