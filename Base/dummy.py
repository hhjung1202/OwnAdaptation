
class Discriminator_target(nn.Module);
    def __init__(self, x_dim = 3, ch_dim = 128):
        super(Discriminator_target, self).__init__()
        self.conv1 = nn.Conv2d(x_dim, ch_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch_dim, ch_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(ch_dim*2)
        self.conv3 = nn.Conv2d(ch_dim*2, ch_dim*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(ch_dim*4)
        self.conv4 = nn.Conv2d(ch_dim*4, ch_dim*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(ch_dim*8)
        self.conv5 = nn.Conv2d(ch_dim*8, 1, kernel_size=4, stride=2, padding=0, bias=False)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


class Generator_target(nn.Module);
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(Generator_target, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(z_dim + y_dim, ch_dim*8 - y_dim, kernel_size=4, stride=1, padding=0)
        self.deconv1_bn = nn.BatchNorm2d(ch_dim*8 - y_dim)
        # concat
        self.deconv2 = nn.ConvTranspose2d(ch_dim*8, ch_dim*4 - y_dim, kernel_size=4, stride=2, padding=1)
        self.deconv2_bn = nn.BatchNorm2d(ch_dim*4 - y_dim)
        # concat
        self.deconv3 = nn.ConvTranspose2d(ch_dim*4, ch_dim*2 - y_dim, kernel_size=4, stride=2, padding=1)
        self.deconv3_bn = nn.BatchNorm2d(ch_dim*2 - y_dim)
        # concat

        self.deconv4 = nn.ConvTranspose2d(ch_dim*2, ch_dim, kernel_size=4, stride=2, padding=1)
        self.deconv4_bn = nn.BatchNorm2d(ch_dim)
        self.deconv5 = nn.ConvTranspose2d(ch_dim, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z, y):
        # x = F.relu(self.deconv1(input))
        yb = y.view(y.size(0), -1, 1, 1)
        z = torch.cat([z.view(-1, 100, 1, 1), yb], 1) # batch, 110, 1, 1

        x = F.relu(self.deconv1_bn(self.deconv1(z)))
        x = conv_y_concat(x, yb)

        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = conv_y_concat(x, yb)

        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = conv_y_concat(x, yb)

        x = F.relu(self.deconv4_bn(self.deconv4(x)))

        x = F.tanh(self.deconv5(x))
        return x

class Classifier_target(nn.Module):
    def __init__(self, num_classes=10, resnet_layer=56):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(50)

        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.bn3  = nn.BatchNorm1d(500)

        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        # mnist
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x