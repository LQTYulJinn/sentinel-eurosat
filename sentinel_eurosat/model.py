import torch.nn as nn


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class Net(nn.Module):
    """Same body for every variant; only the number of input channels changes. Small on purpose: CPU-only training."""
    def __init__(self, in_ch, n_classes=10, w=(16, 32, 64, 128)):
        super().__init__()
        self.f = nn.Sequential(block(in_ch, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3]),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(w[3], n_classes))

    def forward(self, x):
        return self.head(self.f(x))
