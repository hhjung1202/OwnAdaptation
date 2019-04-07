import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# input 32 * 32

def conv_y_concat(x, y):
    x = torch.cat([x,y*torch.ones(x.size(0), y.size(1), x.size(2), x.size(3)).cuda()], 1)
    return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, theta):
        ctx.theta = theta

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.theta

        return output, None


class Discriminator(nn.Module):
    def __init__(self, x_dim = 3, ch_dim = 128): # if MNIST, x_dim = 1; elif SVHN, x_dim = 3;
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(x_dim, ch_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch_dim, ch_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(ch_dim*2)
        self.conv3 = nn.Conv2d(ch_dim*2, ch_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(ch_dim*4)
        self.conv4 = nn.Conv2d(ch_dim*4, ch_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(ch_dim*8)
        self.conv5 = nn.Conv2d(ch_dim*8, 1, kernel_size=4, stride=2, padding=0, bias=False)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x


class Generator(nn.Module):
    def __init__(self, out_dim = 3, z_dim = 100, y_dim = 10, ch_dim = 128): # if MNIST, out_dim = 1; elif SVHN, out_dim = 3;
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(z_dim + y_dim, ch_dim*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(ch_dim*8)
        self.deconv2 = nn.ConvTranspose2d(ch_dim*8, ch_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(ch_dim*4)
        self.deconv3 = nn.ConvTranspose2d(ch_dim*4, ch_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(ch_dim*2)
        self.deconv4 = nn.ConvTranspose2d(ch_dim*2, ch_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4_bn = nn.BatchNorm2d(ch_dim)
        self.deconv5 = nn.ConvTranspose2d(ch_dim, out_dim, 4, 2, 1)

    # forward method
    def forward(self, z, y):
        # x = F.relu(self.deconv1(input))
        yb = y.view(y.size(0), -1, 1, 1)
        z = torch.cat([z.view(-1, 100, 1, 1), yb.float()], 1) # batch, 110, 1, 1, DTYPE ISSUE

        x = F.relu(self.deconv1_bn(self.deconv1(z)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class Discriminator_Dataset(nn.Module): # SVHN RGB Image to Black Image. I think, It will be changed!!!!
    def __init__(self, in_channels, y_dim = 10):
        super(Discriminator_Dataset, self).__init__()

        self.layer1 = Block(in_channels=in_channels, out_channels=in_channels*2, stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc1 = nn.Linear(in_channels*2, y_dim)
        self.fc2 = nn.Linear(y_dim, 1)
        self.LogSoftmax = nn.LogSoftmax()

    # forward method
    def forward(self, x):
        
        x = self.avgpool(self.layer1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

class Classifier(nn.Module):
    def __init__(self, in_channels, y_dim=10):
        super(Classifier, self).__init__()

        self.layer1 = Block(in_channels=in_channels, out_channels=in_channels*2, stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc1 = nn.Linear(in_channels*2, y_dim)

    # forward method
    def forward(self, x):
        
        x = self.avgpool(self.layer1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

class Shared_Model(nn.Module): # SVHN RGB Image to Black Image. I think, It will be changed!!!!
    def __init__(self, x_dim = 3, y_dim = 10):
        super(Shared_Model, self).__init__()

        self.conv1 = nn.Conv2d(x_dim, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = Block(in_channels=16, out_channels=16, stride=1)
        self.layer2 = Block(in_channels=16, out_channels=32, stride=2)
        self.layer3 = Block(in_channels=32, out_channels=64, stride=2)
        self.layer4 = Block(in_channels=64, out_channels=128, stride=2)
        self.Discriminator_Dataset = Discriminator_Dataset(in_channels=128)
        self.Classifier = Classifier(in_channels=128)
        # self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        # self.fc = nn.Linear(128, y_dim)

    def forward(self, x, theta):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        reversed_x = ReverseLayerF.apply(x, theta=theta)
        class_output = self.Classifier(x, y_dim=y_dim)
        domain_output = self.Discriminator_Dataset(reversed_x, y_dim=y_dim)

        return class_output, domain_output