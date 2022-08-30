from operator import truediv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['MyNet']


class MyNet(nn.Module):
    def __init__(self, num_classes=2, backbone_fc=True):
        super(MyNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 1, stride=1, padding=0)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 1, stride=1, padding=0)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.prelu3_2 = nn.PReLU()

        if backbone_fc:
            self.linear = nn.Sequential(
                nn.Linear(128 * 1 * 18, 2),
                nn.PReLU(),
                nn.Linear(2, num_classes)
            )

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 1)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 1)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 1)

        x = x.view(-1, 128 * 1 * 18)
        # for unified style for DFPNet
        out = x.unsqueeze(dim=-1).unsqueeze(dim=-1)

        # return the original feature map if no FC layers.
        if hasattr(self, 'linear'):
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def demo():

    x = torch.randn(1, 1, 1, 18)
    print(x)
    print(x.size())
    net = MyNet(num_classes=2, backbone_fc=False)
    y = net(x)
    print(y.size())

    x = [-1.2611e-03, -5.0107e-03, -5.5262e-04, -4.3407e-03, -5.1200e-05,
    -3.4819e-03, -4.9054e+00,  3.2057e+00,  1.0536e+01,  2.0784e+00,
    1.1622e+01,  3.8205e+00,  1.1497e+02,  3.2958e+01,  7.7813e+01,
    3.1822e+04,  3.2292e+00,  1.3294e+01]
    x = np.array(x)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.reshape(1,1,1,18)
    print(x)
    print(x.size())
    net = MyNet(num_classes=2, backbone_fc=False)
    y = net(x)
    print(y.size())

    net = MyNet(num_classes=2, backbone_fc=True)
    print(net)

#demo()