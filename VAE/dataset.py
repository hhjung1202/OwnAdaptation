import gzip
import os
import pickle
import urllib
import numpy as np
import torch.utils.data as data
from PIL import Image
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



class USPS(data.Dataset):
    """USPS Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        print(img)
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


def usps_loader(img_size, batchSize=128):
    """Get USPS dataset loader."""
    # image pre-processing
    print("USPS Loading")
    pre_process = transforms.Compose([transforms.Resize(img_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5],std=[0.5])])

    # dataset and data loader
    train_dataset = USPS(root='/home/hhjung/hhjung/USPS/',
                        train=True,
                        transform=pre_process,
                        download=True)

    test_dataset = USPS(root='/home/hhjung/hhjung/USPS/',
                        train=False,
                        transform=pre_process,
                        download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batchSize,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batchSize,
        shuffle=True)

    return train_loader, test_loader, 1
