import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(20, 50, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(6 * 6 * 50, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.Softmax()
        )

    def forward(self, x):
        res = self.conv(x)
        res = res.view(res.size(0), -1)
        res = self.linear(res)
        return res
