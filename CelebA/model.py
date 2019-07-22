import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_ch=1, num_classes=4000, dim=20):
        super(Classifier, self).__init__()

        def block(in_ch, out_ch, stride=1):
            layers = [  nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                        nn.MaxPool2d(kernel_size=2), ]
            return layers

        self.model = nn.Sequential(
            *block(input_ch, dim, stride=1),                                # 20, 16, 16
            *block(dim, 2*dim, stride=1),                                   # 40, 8, 8
            *block(2*dim, 3*dim, stride=1),                                 # 60, 4, 4
            nn.Conv2d(3*dim, 4*dim, kernel_size=3, stride=1, padding=1),    # 80, 4, 4
        )

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(4*dim, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x