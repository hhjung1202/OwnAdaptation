import torch
from torchvision import datasets, transforms
import random
from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.ndimage

def rotate(img, degree):
    ori = img.shape
    rot_img = scipy.ndimage.rotate(img, degree, axes=(-2, -1), mode='nearest',)
    return np.resize(rot_img, ori)

# def collate(batch):
#     K = 4
#     each_rotation_degree = 90
#     rot_imgs = []
#     rot_labels = []
#     imgs = []
#     labels = []
#     for x_aug, label in batch:
#         for n in range(K):
#             rot_imgs.append(torch.FloatTensor(rotate(x_aug.numpy(), n*each_rotation_degree).copy()))
#             rot_labels.append(torch.tensor(n))
#         imgs.append(x_aug)
#         labels.append(torch.tensor(label))

#     return [torch.stack(imgs), torch.stack(labels), torch.stack(rot_imgs), torch.stack(rot_labels)]

def cifar100_loader(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("cifar100 Data Loading ...")
    root = '/home/hhjung/hhjung/cifar100/'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
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
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               # collate_fn=collate,
                                               num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              # collate_fn=collate,
                                              num_workers=args.workers)

    return train_loader, test_loader, 3, 100

def cifar10_loader(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("cifar10 Data Loading ...")
    root = '/home/hhjung/hhjung/cifar100/'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='../hhjung/cifar10/',
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='../hhjung/cifar10/',
                                    train=False,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               collate_fn=collate,
                                               num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate,
                                              num_workers=args.workers)

    return train_loader, test_loader, 3, 100

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
        targets[mask] = rnd_targets
        targets = [int(target) for target in targets]
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
        self.targets = targets

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

# def Cifar10_loader(args):
    
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
    
#     print("Cifar10 Data Loading ...")
#     root = '/home/hhjung/hhjung/cifar10/'
    
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
#                              std=(0.2471, 0.2436, 0.2616))
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
#                              std=(0.2471, 0.2436, 0.2616))
#     ])
    
#     # Baseline result
#     Train_dataset = cifar10(noise_type=args.noise_type, noise_rate=args.noise_rate, seed=args.seed
#                                     , root=root, train=True, transform=transform_train, download=True)
#     Test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform_test, download=True)

#     Train_loader = torch.utils.data.DataLoader(dataset=Train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
#     Test_loader = torch.utils.data.DataLoader(dataset=Test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

#     return Train_loader, Test_loader, 3, 10
#     return Train_loader, Test_loader, 3, 10

def Semi_Cifar10_dataset(args):

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
