import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet2_mnist(nn.Module):
    def __init__(self, in_channels, init='random'):
        super(ConvNet2_mnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.linear = nn.Linear(576, 10)
        
        if init == 'gaussian':
            self._gaussian_initialization()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def _gaussian_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.normal_(m.bias, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.normal_(m.bias, mean=0, std=0.01)


class ConvNet3_mnist(nn.Module):
    def __init__(self, in_channels):
        super(ConvNet3_mnist, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(288, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
        