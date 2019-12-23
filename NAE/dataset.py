import torch
from torchvision import datasets, transforms
import random
from PIL import Image
import numpy as np

def MNIST_loader(args):
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    print("MNIST Data Loading ...")
    root = '/home/hhjung/hhjung/MNIST/'
    transform = transforms.Compose([
                                        transforms.Resize(args.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(0.5,)),
                                  ])
    # Baseline result
    Train_dataset = datasets.MNIST(root=root, train=True, transform=transform
                    , target_transform=Symetric_Noise(args.noise_rate, args.sym), download=True)


    Test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)

    Train_loader = torch.utils.data.DataLoader(dataset=Train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    Test_loader = torch.utils.data.DataLoader(dataset=Test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return Train_loader, Test_loader, 1, 10


class Symetric_Noise:
    def __init__(self, probablity=.1, sym=True):
        self.prob = probablity
        self.sym = sym

    def __call__(self, x):
        # Symmetric
        if self.sym:
            item = torch.randint(0, 9, size=[])
            if self.prob >= np.random.rand(1):
                return torch.tensor(item), torch.tensor(x)
            else:
                return torch.tensor(x), torch.tensor(x)

        # ASymmetric
        else:
            if self.prob >= np.random.rand(1):
                if x == 9:
                    return torch.tensor(1), torch.tensor(x)
                    # bird -> airplane
                elif x == 2:
                    return torch.tensor(0), torch.tensor(x)
                    # cat -> dog
                elif x == 3:
                    return torch.tensor(5), torch.tensor(x)
                    # dog -> cat
                elif x == 5:
                    return torch.tensor(3), torch.tensor(x)
                    # deer -> horse
                elif x == 4:
                    return torch.tensor(7), torch.tensor(x)
                else:
                    return torch.tensor(x), torch.tensor(x)
            else:
                return torch.tensor(x), torch.tensor(x)


def Cifar10_loader(args):
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    print("Cifar10 Data Loading ...")
    root = '/home/hhjung/hhjung/cifar10/'
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])
    
    # Baseline result
    Train_dataset = datasets.CIFAR10(root, train=True, transform=transform_train
                    , target_transform=Symetric_Noise(args.noise_rate, args.sym), download=True)
    Test_dataset = datasets.CIFAR10(root, train=False, transform=transform_test, download=True)

    print(Train_dataset[0])
    Train_loader = torch.utils.data.DataLoader(dataset=Train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    Test_loader = torch.utils.data.DataLoader(dataset=Test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return Train_loader, Test_loader, 3, 10

if __name__=='__main__':
    class e():
        pass

    args = e()

    args.batch_size = 128
    args.workers = 4
    args.img_size = 32
    args.noise_rate = 0.1
    args.sym = True
    args.seed = 1234

    Train_loader, Test_loader, i,j = MNIST_loader(args)
    for i, (x, (n, l)) in enumerate(Train_loader):
        print(i)
        print(x)
        print(x.size())
        print(n)
        print(l)
        print()