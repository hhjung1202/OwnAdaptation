import torch
from torchvision import datasets, transforms
import random
from PIL import Image
import numpy as np

class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False, noise_rate=0.1, sample=5000, seed=1234, Task='True'):
        super(MNIST, self).__init__(root, train=train
            , transform=transform, target_transform=None, download=download)

        self.sample = sample
        self.noise_rate = noise_rate
        self.Task = Task

        self.data_shuffle = list(zip(self.data, self.targets))

        random.seed(seed)
        random.shuffle(self.data_shuffle)

        if self.train is True:
            self.Set = self.data_shuffle
        else:
            self.Set = self.data_shuffle

    def __getitem__(self, index):

        if self.train is True:
            img, real_target = self.Set[index]
            noise_sample = self.noise_rate * len(self.data_shuffle)
            if index < noise_sample:
                target = self.Intended_Random_Noise_Label(real_target)
            else:
                target = real_target
        else:
            img, real_target = self.Set[index]
            target = real_target
            
        target = int(target)
        real_target = int(real_target)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, real_target

    def Intended_Random_Noise_Label(self, label):
        item = torch.randint(0,9, size=label.size(), dtype=label.dtype)
        if item >= label:
            return item + 1
        else:
            return item

    def __len__(self):
        """Return the number of images."""
        return len(self.Set)

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
    Train_dataset = MNIST(root=root, train=True, transform=transform, download=True, 
                                        noise_rate=args.noise_rate, sample=args.sample, seed=args.seed)


    Test_dataset = MNIST(root=root, train=False, transform=transform, download=True)

    Train_loader = torch.utils.data.DataLoader(dataset=Train_dataset, batch_size=args.batch_size, shuffle=True)
    Test_loader = torch.utils.data.DataLoader(dataset=Test_dataset, batch_size=args.batch_size, shuffle=False)
    return Train_loader, Test_loader, 1, 10


class cifar10(datasets.CIFAR10):
    def __init__(self, noise_type='sym', noise_rate=0.0, seed=1234, **kwargs):
        super(cifar10, self).__init__(**kwargs)
        self.seed = seed
        self.num_classes = 10
        self.flip_pairs = np.asarray([[9, 1], [2, 0], [4, 7], [3, 5], [5, 3]])

        self.real_target = self.targets

        if noise_rate > 0:
            if noise_type == 'sym':
                self.symmetric_noise(noise_rate)
            elif noise_type == 'Asym':
                self.asymmetric_noise(noise_rate)
            else:
                raise ValueError(
                    'expected noise_type is either sym or Asym '
                    '(got {})'.format(noise_type))

        self.data_zip = list(zip(self.data, self.targets, self.real_target))


    def symmetric_noise(self, noise_rate):
        
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        mask = np.random.rand(len(targets)) <= noise_rate
        rnd_targets = np.random.choice(self.num_classes, mask.sum())
        print(rnd_targets)
        targets[mask] = rnd_targets
        print(targets[mask])
        targets = [int(target) for target in targets]
        print(targets)
        print(self.targets)
        self.targets = targets

    def asymmetric_noise(self, noise_rate):
        
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        for i, target in enumerate(targets):
            if target in self.flip_pairs[:, 0]:
                if np.random.uniform(0, 1) <= noise_rate:
                    idx = int(np.where(self.flip_pairs[:, 0] == target)[0])
                    targets[i] = self.flip_pairs[idx, 1]
        targets = [int(x) for x in targets]
        self.targets = target_transform

    def __getitem__(self, index):

        img, target, real_target = self.data_zip[index]
            
        target = int(target)
        real_target = int(real_target)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, real_target

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
    Train_dataset = cifar10(noise_type=args.noise_type, noise_rate=args.noise_rate, seed=args.seed
                                    , root=root, train=True, transform=transform_train, download=True)
    Test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform_test, download=True)

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
    args.noise_type = "sym"
    args.sym = True
    args.seed = 1234

    Train_loader, Test_loader, i,j = Cifar10_loader(args)
    for i, (x, n, l) in enumerate(Train_loader):
        print(i)
        print(x)
        print(x.size())
        print(n)
        print(l)
        print()