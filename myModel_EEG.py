import torch
import torch.nn as nn
from thop import profile


class ReLUConvBn(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBn, self).__init__()
        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)


class OneD_CNN(nn.Module):
    def __init__(self, input_size, channel):
        super(OneD_CNN, self).__init__()

        self.net = nn.Sequential(
            ReLUConvBn(channel, channel * 2, kernel_size=9, stride=4, padding=0),
            ReLUConvBn(channel * 2, channel * 2, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(channel * 2, channel * 2, kernel_size=3, stride=1, padding=0),

            ReLUConvBn(channel * 2, channel * 4, kernel_size=9, stride=4, padding=0),
            ReLUConvBn(channel * 4, channel * 4, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(channel * 4, channel * 4, kernel_size=3, stride=1, padding=0),

            # ReLUConvBn(channel*4,channel*8,kernel_size=3,stride=2,padding=0),
            # ReLUConvBn(channel*8,channel*8,kernel_size=3,stride=2,padding=0),
            # ReLUConvBn(channel*8,channel*8,kernel_size=3,stride=2,padding=0),

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


class ResDepSepBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, ratio=6):
        super(ResDepSepBlock, self).__init__()
        hidden_dim = C_in * ratio
        padding = int(kernel_size / 2)
        self.use_res_connect = stride == 1 and C_in == C_out

        self.net = nn.Sequential(
            nn.Conv1d(C_in, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv1d(hidden_dim, C_out, 1, bias=False),
            nn.BatchNorm1d(C_out),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.net(x)
        else:
            return self.net(x)


class OneD_ResCNN(nn.Module):
    def __init__(self, input_size, channel):
        super(OneD_ResCNN, self).__init__()

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
            nn.ReLU6(inplace=False),
            nn.Linear(channel * 1, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.squeeze(x)
        return self.out(x), x


if __name__ == '__main__':
    from thop import profile

    model = OneD_CNN(600, 30)
    x1 = torch.randn(1, 1, 30, 600)
    x2 = torch.randn(2, 36, 30)
    x3 = torch.randn(2, 36, 30)
    flops, params = profile(model, inputs=(x1))
    print(flops, params)