import torch
from torchvision import datasets, transforms
import random
from PIL import Image
import numpy as np
import torch.utils.data as data

def Semi_Cifar10_dataset(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    print("Cifar10 Data Loading ...")
    root = '/home/hhjung/hhjung/cifar10/'
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        
    ])
    

    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, int(args.n_labeled/10))

    train_labeled_dataset = CIFAR10_labeled(root, None, train=True, transform=transform_train)
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=transform_train)
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_test, download=True)

    print ("#Labeled: {} #Unlabeled: {}".format(len(train_labeled_idxs), len(train_unlabeled_idxs)))
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
    

def train_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs


class CIFAR10_labeled(datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])

if __name__=='__main__':
    class e():
        pass

    args = e()

    args.batch_size = 128
    args.workers = 4
    args.img_size = 32
    args.noise_rate = 0.1
    args.Anchor = 2
    args.noise_type = "Asym"
    args.sym = True
    args.seed = 1234

    Sample_loader = Cifar10_Sample(args)
    AnchorSet = iter(Sample_loader).next()
    x, l = AnchorSet
    print(x, l)
    # for i, (x, l) in enumerate(Sample_loader):
        # print(l)
