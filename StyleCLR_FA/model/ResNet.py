import torch
import torch.nn as nn
import torch.nn.functional as F
from .Feature_Adaptive import *
from .Contrastive_Loss import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module):
    def __init__(self, blocks, num_blocks, style_out, num_classes, n, L_type):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.style_out = style_out
        self.n = n
        self.L_type = L_type
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
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.f_x = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
        self.Content_Contrastive = Content_Contrastive(temperature=1.)
        self.Style_Contrastive = Style_Contrastive()
        self.Semi_Loss = Semi_Loss(temperature=1.)

    def forward(self, x, y, u_x, test=False):
        if test:
            return self.test_(x), self.test_style(x)

        x_ = torch.cat([x, u_x], dim=0)
        x_ = self.init(x_)

        (b, c, w, h), size_s = x_.size(), x.size(0)
        n = self.n if b >= self.n else b-1
        x_ = torch.cat([x_.repeat(1, n, 1, 1).view(b*n, c, w, h), x_], 0) # AAA BBB CCC ABC
        style_label = self.LongTensor(self.style_gen(b, n))
        style_loss = None

        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x_ = layer(x_, style_label, b)

            if i is 3: # 1 3 5 7
                style_loss = self.forward_style(x_, style_label, b, n)

        if style_loss is None:
            style_loss = self.forward_style(x_, style_label, b, n)
        content_loss = self.forward_content(x_, b, n)
        loss_s, JS_loss, loss_u = self.forward_classifier(x_, b, n, size_s, y)

        return loss_s, JS_loss, loss_u, style_loss, content_loss

    def forward_style(self, x, style_label, b, n):
        x = self.g_x(x)
        content = x[:-b]
        style = x[-b:]
        style_loss = self.Style_Contrastive(content, style, style_label, b, n, L_type=self.L_type)

        return style_loss

    def forward_content(self, x, b, n):
        x = self.flatten(self.avgpool(x))
        x = self.f_x(x)
        content = x[:-b]
        style = x[-b:]
        content_loss = self.Content_Contrastive(content, style, b, n)

        return content_loss

    def forward_classifier(self, x, b, n, size_s, y):
        
        x = self.flatten(self.avgpool(x))
        logits = self.linear(x)
        loss_s, JS_loss, loss_u = self.Semi_Loss(logits, b, n, size_s, y)

        return loss_s, JS_loss, loss_u


    def style_gen(self, batch_size, n):
        i = torch.randint(1, batch_size, (1,))[0]
        perm = [_ for _ in range(batch_size)]
        style_label = []
        for m in range(batch_size):
            for k in range(n):
                style_label.append((perm[(i+m+k)%batch_size]))

        return style_label

    def test_style(self, x):

        x_ = self.init(x)

        b, c, w, h = x_.size()
        n = self.n if b >= self.n else b-1
        x_ = torch.cat([x_.repeat(1, n, 1, 1).view(b*n, c, w, h), x_], 0) # AAA BBB CCC ABC
        style_label = self.style_gen(b, n)

        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x_ = layer(x_, style_label, b)

        x_ = self.flatten(self.avgpool(x_[:-b]))
        out = self.linear(x_)

        return out

    def test_(self, x):

        x_ = self.init(x)

        b, c, w, h = x_.size()
        x_ = torch.cat([x_, x_], 0) # AAA BBB CCC ABC
        style_label = [i for i in range(b)]

        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x_ = layer(x_, style_label, b)

        x_ = self.flatten(self.avgpool(x_[:-b]))
        out = self.linear(x_)

        return out


def ResNet18(serial=[0,0,0,0,0,0,0,0], style_out=0, num_blocks=[2,2,2,2], num_classes=10, n=4, L_type='c1'):
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

    return ResNet(blocks, num_blocks, style_out=style_out, num_classes=num_classes, n=n, L_type=L_type)

def ResNet34(serial=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], style_out=0, num_blocks=[3,4,6,3], num_classes=10, n=4, L_type='c1'):
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

    return ResNet(blocks, num_blocks, style_out=style_out, num_classes=num_classes, n=n, L_type=L_type)
