import torch
from torchvision import datasets, transforms

def SVHN_loader(img_size, batchSize=128):
    print("SVHN Data Loading ...")

    rgb2grayWeights = [0.2989, 0.5870, 0.1140]
    train_dataset = datasets.SVHN(root='/home/hhjung/hhjung/SVHN/', split='train', 
                                        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                        download=True)

    test_dataset = datasets.SVHN(root='/home/hhjung/hhjung/SVHN/', split='test', 
                                        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                        download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)
    return train_loader, test_loader, 3

    
def MNIST_loader(img_size, batchSize=128):
    print("MNIST Data Loading ...")

    train_dataset = datasets.MNIST(root='/home/hhjung/hhjung/MNIST/', train=True,
                                        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                        download=True)

    test_dataset = datasets.MNIST(root='/home/hhjung/hhjung/MNIST/', train=False,
                                       transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()
                                            , transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                       download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)
    return train_loader, test_loader, 1


def cifar10_loader():
    batch_size = 128
    rgb2grayWeights = [0.2989, 0.5870, 0.1140]

    print("cifar10 Data Loading ...")
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

    train_dataset = datasets.CIFAR10(root='../hhjung/cifar10/', train=True, transform=transform_train, download=True)

    test_dataset = datasets.CIFAR10(root='../hhjung/cifar10/', train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def cifar100_loader():
    batch_size = 128
    print("cifar100 Data Loading ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(root='../hhjung/cifar100/',
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    test_dataset = datasets.CIFAR100(root='../hhjung/cifar100/',
                                    train=False,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    return train_loader, test_loader