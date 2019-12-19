import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, size=1):
        super(UnFlatten, self).__init__()
        self.size = size
    def forward(self, x):
        return x.view(x.size(0), -1, self.size, self.size)

class Block_A(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(Block_A, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels) #
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        out = F.relu(self.bn1(x))
        h1 = self.conv1(out)
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h2 = self.conv3(out)
        return h1+h2

class Block_B(nn.Module):

    def __init__(self, in_channels):
        super(Block_B, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels) #
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        return h + x


class PreActResNet(nn.Module):
    def __init__(self, num_classes=10, resnet_layer=32, memory=0):
        super(ResNet, self).__init__()

        filters = [32, 32, 64, 128]
        self.memory = memory
        self.conv = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=False)

        if resnet_layer is 14:
            self.n = 2
        elif resnet_layer is 20:
            self.n = 3
        elif resnet_layer is 32:
            self.n = 5
        elif resnet_layer is 44:
            self.n = 7
        elif resnet_layer is 56:
            self.n = 9
        elif resnet_layer is 110:
            self.n = 18

        self._forward = ["conv"]
            
        setattr(self, 'layer1_0', Block_A(in_channels=filters[0], out_channels=filters[1], stride=1))
        self._forward.append('layer1_0')
        for i in range(1,self.n):
            setattr(self, 'layer1_%d' % (i), Block_B(in_channels=filters[1]))
            self._forward.append('layer1_%d' % (i))

        setattr(self, 'layer2_0', Block_A(in_channels=filters[0], out_channels=filters[1], stride=1))
        self._forward.append('layer2_0')
        for i in range(1,self.n):
            setattr(self, 'layer2_%d' % (i), Block_B(in_channels=filters[1]))
            self._forward.append('layer2_%d' % (i))

        setattr(self, 'layer3_0', Block_A(in_channels=filters[0], out_channels=filters[1], stride=1))
        self._forward.append('layer3_0')
        for i in range(1,self.n):
            setattr(self, 'layer3_%d' % (i), Block_B(in_channels=filters[1]))
            self._forward.append('layer3_%d' % (i))

        self.bn = nn.BatchNorm2d(filters[3])
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(filters[3], num_classes)

    def forward(self, x):
        c = None
        for i, name in enumerate(self._forward):
            layer = getattr(self, name)
            x = layer(x)
            if i == self.memory:
                c = x

        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if c is None:
            c = x

        x = self.fc(x)
        return x, c

class PreActResNet(chainer.Chain):

    def __init__(self, layer_names=None):
        super().__init__()
        kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        block = [5, 5, 5]
        filters = [32, 32, 64, 128]

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, filters[0], 3, 1, 1, **kwargs, nobias=True)
            self.res2 = BuildingBlock(block[0], filters[0], filters[1], 1, **kwargs)
            self.res3 = BuildingBlock(block[1], filters[1], filters[2], 2, **kwargs)
            self.res4 = BuildingBlock(block[2], filters[2], filters[3], 2, **kwargs)
            self.bn4 = L.BatchNormalization(filters[3])
            self.fc5 = L.Linear(filters[3], 10)

        self.functions = collections.OrderedDict([
            ('conv1', [self.conv1]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4, self.bn4, F.relu]),
            ('pool4', [R._global_average_pooling_2d]),
            ('fc5', [self.fc5]),
        ])
        if layer_names is None:
            layer_names = list(self.functions.keys())[-1]
        if (not isinstance(layer_names, str) and
                all([isinstance(name, str) for name in layer_names])):
            return_tuple = True
        else:
            return_tuple = False
            layer_names = [layer_names]
        self._return_tuple = return_tuple
        self._layer_names = layer_names

    def __call__(self, x):
        h = x

        activations = dict()
        target_layers = set(self._layer_names)
        for key, funcs in self.functions.items():
            if len(target_layers) == 0:
                break
            for func in funcs:
                h = func(h)
            if key in target_layers:
                activations[key] = h
                target_layers.remove(key)

        if self._return_tuple:
            activations = tuple(
                [activations[name] for name in self._layer_names])
        else:
            activations = list(activations.values())[0]

        return activations












class Encoder(nn.Module):
    def __init__(self, I=32, H=400, latent_size=64):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(I*I, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, latent_size),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class Classifier(nn.Module):
    def __init__(self, latent_size=64, num_classes=10):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, num_classes),
        )

    def forward(self, z):
        z = self.model(z)
        return z

class NAE(nn.Module):
    def __init__(self, I=32, H=400, latent_size=64, num_classes=10):
        super(NAE, self).__init__()

        self.Encoder = Encoder(I=I, H=H, latent_size=latent_size)
        self.Classifier = Classifier(latent_size=latent_size, num_classes=num_classes)

    def forward(self, x, test=False):
        z = self.Encoder(x)
        c = self.Classifier(z)
        return z, c





# class Encoder(nn.Module):
#     def __init__(self, chIn=1, feature=64):
#         super(Encoder, self).__init__()

#         self.model = nn.Sequential(
#             nn.Conv2d(chIn, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),

#             nn.AvgPool2d(kernel_size=4, stride=4),
#             Flatten(),
#             nn.Linear(256, feature),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature, feature),
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, chIn=1, feature=64):
#         super(Decoder, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(feature, feature),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature, 256),
#             UnFlatten(size=2)   # 2*2*64

#             nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1) # 4*4*64
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1) # 8*8*32
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1) # 16*16*16
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(16, chIn, 3, stride=2, padding=1, output_padding=1) # 32*32*1
#             nn.Tanh(),
#         )
        
#     def forward(self, x):
#         x = self.model(x)
#         return x