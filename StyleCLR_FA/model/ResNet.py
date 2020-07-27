import torch
import torch.nn as nn
import torch.nn.functional as F
from .Feature_Adaptive import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNet(nn.Module):
    def __init__(self, blocks, num_blocks, style_out, num_classes=10, n=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.style_out = style_out
        self.n = n

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
        # self.L1Loss = torch.nn.L1Loss()
        self.MSELoss = nn.MSELoss()
        self.softmin = nn.Softmin(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.CE = nn.CrossEntropyLoss()

        self.g_x = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x, u_x):

        x_ = torch.cat([x, u_x], dim=0)
        x_ = self.init(x_)

        b, c, w, h = x_.size()
        n = self.n
        if b < n: n = b-1;
        x_ = torch.cat([x_.repeat(1, n, 1, 1).view(b*n, c, w, h), x_], 0) # AAA BBB CCC ABC
        style_label = self.style_gen(b, n)

        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x_ = layer(x_, style_label, b)

            if i+1 is self.style_out: # 2, 4, 6, 8
                style = x_[-b:]
                content_feat = x_[:-b]
        
        style_loss = self.style_contrastive(x_[:-b], x_[-b:], style_label, b, n) # x_[:-b], x[-b:] is Content, Style
        # style_loss = self.style_reconstruction(x_[:-b], x_[-b:], style_label)
        x = self.flatten(self.avgpool(content_feat))
        x = self.linear(x)

        return x, style_loss

    def style_gen(self, batch_size, n):
        i = torch.randint(1, batch_size, (1,))[0]
        perm = [_ for _ in range(batch_size)]
        arr = []
        for m in range(batch_size):
            for k in range(n):
                arr.append((perm[(i+m+k)%batch_size]))

        self.style_label = arr


    def style_contrastive(self, content, style, style_label, b, n):
        f_c = gram_matrix(content).view(n,b,-1)             # n, b, ch * ch
        f_s = gram_matrix(style)[style_label].view(n,b,-1)  # n, b, ch * ch

        f_c = f_c.transpose(0,1).repeat(1, n, 1)                    # b, n*n, -1 AAA_ -> AAA_ AAA_ AAA_
        f_s = f_s.transpose(0,1).repeat(1, 1, n).view(b, n*n, -1)   # b, n*n, -1 BCD -> BBB CCC DDD

        mse = ((f_c - f_s)**2).sum(dim=2).view(b*n,n)

        # case 1
        style_loss = self.softmin_ce(mse, style_label)

        # case 2
        # style_loss = self.softmax_ce_rev(mse, style_label)

        return style_loss

    def style_reconstruction(self, content, style, style_label):
        f_c = gram_matrix(content) # b*n, ch, ch
        f_s = gram_matrix(style) # b, ch, ch
        adaptive_s = f_s[style_label] # b*n, ch, ch
        style_loss = self.MSELoss(f_c, adaptive_s)
        return style_loss


    def softmin_ce(self, input, target): # y * log(p), p = softmax(-out)
        log_likelihood = self.softmin(input).log()
        nll_loss = F.nll_loss(log_likelihood, target)
        return nll_loss

    def softmax_ce_rev(self, input, target): # y * log(1-p), p = softmax(out)
        log_likelihood_reverse = torch.log(1 - self.softmax(input))
        nll_loss = F.nll_loss(log_likelihood_reverse, target)
        return nll_loss





def ResNet18(serial=[0,0,0,0,0,0,0,0], style_out=0, num_blocks=[2,2,2,2], num_classes=10):
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

    return ResNet(blocks, num_blocks, style_out=style_out, num_classes=num_classes)

def ResNet34(serial=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], style_out=0, num_blocks=[3,4,6,3], num_classes=10):
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

    return ResNet(blocks, num_blocks, style_out=style_out, num_classes=num_classes)


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a, b, c * d)
    G = torch.bmm(features, torch.transpose(features, 1,2))
    return G.div(b * c * d)

# def gram_matrix2(input):
#     a, b, c, d = input.size()
#     features = input.view(a, b, c * d)
#     G = torch.bmm(torch.transpose(features, 1,2), features)
#     return G.div(b * c * d)