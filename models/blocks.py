import torch
import torch.nn as nn


class ReLUConvBn(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBn, self).__init__()
        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

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