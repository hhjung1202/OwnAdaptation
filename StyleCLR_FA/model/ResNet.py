import torch
import torch.nn as nn
import torch.nn.functional as F
from .Feature_Adaptive import *
from .Contrastive_Loss import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResNet(nn.Module):
    def __init__(self, blocks, num_blocks, style_out, num_classes, n):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.style_out = style_out
        self.n = n
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

        index = 0

        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._forward = []

        setattr(self, 'layer1_0', blocks[index](64, 64, 1)); index+=1
        self._forward.append('layer1_0')
        for i in range(1, num_blocks[0]):
            setattr(self, 'layer1_%d' % (i), blocks[index](64, 64)); index+=1
            self._forward.append('layer1_%d' % (i))

        setattr(self, 'layer2_0', blocks[index](64, 128, 2)); index+=1
        self._forward.append('layer2_0')
        for i in range(1, num_blocks[1]):
            setattr(self, 'layer2_%d' % (i), blocks[index](128, 128)); index+=1
            self._forward.append('layer2_%d' % (i))

        setattr(self, 'layer3_0', blocks[index](128, 256, 2)); index+=1
        self._forward.append('layer3_0')
        for i in range(1, num_blocks[2]):
            setattr(self, 'layer3_%d' % (i), blocks[index](256, 256)); index+=1
            self._forward.append('layer3_%d' % (i))

        setattr(self, 'layer4_0', blocks[index](256, 512, 2)); index+=1
        self._forward.append('layer4_0')
        for i in range(1, num_blocks[3]):
            setattr(self, 'layer4_%d' % (i), blocks[index](512, 512)); index+=1
            self._forward.append('layer4_%d' % (i))

        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = Flatten()

        self.g_x = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.f_x = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
        self.Content_Contrastive = Content_Contrastive(temperature=1.)
        self.Style_Contrastive = Style_Contrastive()

    def forward(self, x, t="self"):
        if t=="cls":
            x = self.init(x)
            for i, name in enumerate(self._forward):
                layer = getattr(self, name)
                x = layer(x, None, None)
            # content_loss = self.forward_content(x_, b, n)
            x = self.flatten(self.avgpool(x))
            x = self.linear(x)
            return x, None, None
        x = self.init(x)
        b, c, w, h = x.size()
        n = self.n if b >= self.n else b-1
        x = torch.cat([x.repeat(1, n, 1, 1).view(b*n, c, w, h), x], 0) # AAA BBB CCC ABC
        style_label = self.LongTensor(self.style_gen(b, n))

        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x = layer(x, style_label, b)

        st_mse, st_label = self.forward_style(x, style_label, b, n)
        # content_loss = self.forward_content(x_, b, n)
        logits = self.flatten(self.avgpool(x[-b:]))
        logits = self.linear(logits)

        return logits, st_mse, st_label

    def forward_style(self, x, style_label, b, n):
        x = self.g_x(x)
        content = x[:-b]
        style = x[-b:]
        st_mse, st_label = self.Style_Contrastive(content, style, style_label, b, n)

        return st_mse, st_label

    # def forward_content(self, x, b, n):
    #     x = self.flatten(self.avgpool(x))
    #     x = self.f_x(x)
    #     content = x[:-b]
    #     style = x[-b:]
    #     content_loss = self.Content_Contrastive(content, style, b, n)

    #     return content_loss

    def forward_classifier(self, x):
        
        x = self.flatten(self.avgpool(x))
        logits = self.linear(x)
        return logits

    def style_gen(self, batch_size, n):
        i = torch.randint(1, batch_size, (1,))[0]
        perm = [_ for _ in range(batch_size)]
        style_label = []
        for m in range(batch_size):
            for k in range(n):
                style_label.append((perm[(i+m+k)%batch_size]))

        return style_label

def ResNet18(serial=[0,0,0,0,0,0,0,0], style_out=0, num_blocks=[2,2,2,2], num_classes=10, n=4):
    blocks = []
    for i in range(8):
        if serial[i] is 0:
            blocks.append(BasicBlock)
        elif serial[i] is 1:
            blocks.append(AdaptiveBlock)
        elif serial[i] is 2:
            blocks.append(PreBlock)
        elif serial[i] is 3:
            blocks.append(PostBlock)
        elif serial[i] is 4:
            blocks.append(InstanceBlock)

    return ResNet(blocks, num_blocks, style_out=style_out, num_classes=num_classes, n=n)

def ResNet34(serial=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], style_out=0, num_blocks=[3,4,6,3], num_classes=10, n=4):
    blocks = []
    for i in range(8):
        if serial[i] is 0:
            blocks.append(BasicBlock)
        elif serial[i] is 1:
            blocks.append(AdaptiveBlock)
        elif serial[i] is 2:
            blocks.append(PreBlock)
        elif serial[i] is 3:
            blocks.append(PostBlock)
        elif serial[i] is 4:
            blocks.append(InstanceBlock)

    return ResNet(blocks, num_blocks, style_out=style_out, num_classes=num_classes, n=n)
