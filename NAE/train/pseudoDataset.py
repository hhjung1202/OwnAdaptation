import torch
from torchvision import datasets, transforms
from PIL import Image

class cifar10_pseudo(datasets.CIFAR10):
    def __init__(self, label, **kwargs):
        super(cifar10_pseudo, self).__init__(**kwargs)
        self.num_classes = 10
        self.real_target = self.targets
        self.data_zip = list(zip(self.data, label.numpy(), self.real_target))

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

def Cifar10_temp_loader(args):
    
    print("Cifar10 Temp Data Loading ...")
    root = '/home/hhjung/hhjung/cifar10/'
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    # Baseline result
    Train_dataset = datasets.CIFAR10(root=root, train=True, transform=transform_train, download=True)

    Temp_loader = torch.utils.data.DataLoader(dataset=Train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return Temp_loader

def Cifar10_pseudo_loader(args, Pseudo_label):
    
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

    # Baseline result
    Train_dataset = cifar10_pseudo(label=Pseudo_label, root=root, train=True, transform=transform_train, download=True)

    Pseudo_loader = torch.utils.data.DataLoader(dataset=Train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    return Pseudo_loader

if __name__=='__main__':
    class e():
        pass

    args = e()

    args.batch_size = 128
    args.workers = 4
    args.img_size = 32
    args.noise_rate = 0.1
    args.noise_type = "Asym"
    args.sym = True
    args.seed = 1234

    item = torch.randint(0,9, size=[50000], dtype=torch.int64)
    print(item.size())

    Train_loader = Cifar10_pseudo_loader(args, item)
    for i, (x, n, l) in enumerate(Train_loader):
        print(torch.sum(n.eq(l)))
