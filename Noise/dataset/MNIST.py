import torch
from torchvision import datasets, transforms
import random
from PIL import Image


class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False, noise_rate=0.1, dist_rate=0.1, sample=5000, seed=1234, Task='True'):
        super(MNIST, self).__init__(root, train=train
            , transform=transform, target_transform=None, download=download)

        self.sample = sample
        self.noise_rate = noise_rate
        self.dist_rate = dist_rate
        self.Task = Task

        self.data_shuffle = list(zip(self.data, self.targets))

        random.seed(seed)
        random.shuffle(self.data_shuffle)

        if self.train is True:

            if self.Task == "True": # 5000 True Set
                self.Set = self.data_shuffle[:self.sample]
            elif self.Task == "Fake": # 5000 Random Labeled Set
                self.Set = self.data_shuffle[self.sample:2*self.sample]
            elif self.Task == "Noise": # 50000 N% Noise labeled Set [5000 True Set(Known), N% Noise labeled Set, Left True Set(Unknown)]
                self.CleanSet = self.data_shuffle[:self.sample]
                self.NoiseSet = self.data_shuffle[self.sample:]
                random.shuffle(self.NoiseSet)
                self.Set = self.CleanSet + self.NoiseSet
            elif self.Task == "Noise_Test": # 45000 N% Noise labeled Set [N% Noise labeled Set]
                self.NoiseSet = self.data_shuffle[self.sample:]
                random.shuffle(self.NoiseSet)
                self.Set = self.NoiseSet
            elif self.Task == "Noise_Sample": # 45000 N% Noise labeled Set [N% Noise labeled Set]
                self.Set = self.data_shuffle[:self.sample]
            elif self.Task == "Noise_Triple": # All N% Noise labeled Set
                self.Set = self.data_shuffle
            else: # All N% Noise labeled Set [without True Set, Only Noisy label]
                self.Set = self.data_shuffle
        else:
            self.Set = self.data_shuffle

    def __getitem__(self, index):

        if self.train is True:
            if self.Task == "True": # 5000 True Set
                img, real_target = self.Set[index]
                target = real_target

            elif self.Task == "Fake": # 5000 Random Labeled Set
                img, real_target = self.Set[index]
                target = torch.randint_like(real_target, low=0, high=10)

            elif self.Task == "Noise": # 50000 N% Noise labeled Set [5000 True Set(Known), N% Noise labeled Set, Left True Set(Unknown)]
                img, real_target = self.Set[index]
                noise_sample = int(self.noise_rate * len(self.data_shuffle))
                if self.sample <= index and index < self.sample + noise_sample:
                    target = self.Intended_Random_Noise_Label(real_target)
                else:
                    target = real_target

            elif self.Task == "Noise_Test": # 45000 N% Noise labeled Set [N% Noise labeled Set]
                img, real_target = self.Set[index]
                noise_sample = int(0.5 * len(self.Set))
                if index < noise_sample:
                    target = self.Intended_Random_Noise_Label(real_target)
                else:
                    target = real_target

            elif self.Task == "Noise_Sample": # 5000 N% Noise labeled Set [N% Noise labeled Set]
                img, real_target = self.Set[index]
                noise_sample = int(self.noise_rate * len(self.Set))
                dist_sample = int(self.dist_rate * len(self.Set))
                if index < noise_sample:
                    target = self.Intended_Random_Noise_Label(real_target)
                else:
                    target = real_target

                if index < dist_sample//2:
                    real_target = target

                if index >= len(self.Set) - dist_sample//2:
                    real_target = self.Intended_Random_Noise_Label(real_target)

            elif self.Task == "Noise_Triple": # All N% Noise labeled Set
                img, real_target = self.Set[index]
                noise_sample = int(self.noise_rate * len(self.Set))
                if index < noise_sample:
                    target = self.Intended_Random_Noise_Label(real_target)
                else:
                    target = real_target

            else: # All Data
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
    Task = ["True", "Fake", "Noise", "Noise_Test", "All", "Noise_Sample", "Noise_Triple"]

    # Discriminator Method for True
    True_dataset = MNIST(root=root, train=True, transform=transform, download=True, 
                                        noise_rate=args.noise_rate, sample=args.sample, seed=args.seed, 
                                        Task=Task[0])

    # Discriminator Method for Fake
    Fake_dataset = MNIST(root=root, train=True, transform=transform, download=True, 
                                        noise_rate=args.noise_rate, sample=args.sample, seed=args.seed, 
                                        Task=Task[1])

    # Proposed Method
    Noise_dataset = MNIST(root=root, train=True, transform=transform, download=True, 
                                        noise_rate=args.noise_rate, sample=args.sample, seed=args.seed, 
                                        Task=Task[2])

    # Proposed Method Test
    Noise_Test_dataset = MNIST(root=root, train=True, transform=transform, download=True, 
                                        noise_rate=args.noise_rate, sample=args.sample, seed=args.seed, 
                                        Task=Task[3])

    # Baseline result
    All_dataset = MNIST(root=root, train=True, transform=transform, download=True, 
                                        noise_rate=args.noise_rate, sample=args.sample, seed=args.seed, 
                                        Task=Task[4])

    # 5000 Noise Sample Dataset
    Noise_Sample_dataset = MNIST(root=root, train=True, transform=transform, download=True, 
                                        noise_rate=args.noise_rate, sample=args.sample, seed=args.seed, 
                                        Task=Task[5])

    # 5000 Noise Sample Dataset
    Noise_Triple_dataset = MNIST(root=root, train=True, transform=transform, download=True, 
                                        noise_rate=args.noise_rate, sample=args.sample, seed=args.seed, 
                                        Task=Task[6])


    Test_dataset = MNIST(root=root, train=False, transform=transform, download=True)

    True_loader = torch.utils.data.DataLoader(dataset=True_dataset, batch_size=args.batch_size, shuffle=True)
    Fake_loader = torch.utils.data.DataLoader(dataset=Fake_dataset, batch_size=args.batch_size, shuffle=True)
    Noise_loader = torch.utils.data.DataLoader(dataset=Noise_dataset, batch_size=args.batch_size, shuffle=True)
    Noise_Test_loader = torch.utils.data.DataLoader(dataset=Noise_Test_dataset, batch_size=args.batch_size, shuffle=False)
    Noise_Sample_loader = torch.utils.data.DataLoader(dataset=Noise_Sample_dataset, batch_size=args.batch_size, shuffle=True)
    Noise_Triple_loader = torch.utils.data.DataLoader(dataset=Noise_Triple_dataset, batch_size=args.batch_size, shuffle=True)
    All_loader = torch.utils.data.DataLoader(dataset=All_dataset, batch_size=args.batch_size, shuffle=True)
    Test_loader = torch.utils.data.DataLoader(dataset=Test_dataset, batch_size=args.batch_size, shuffle=False)
    return True_loader, Fake_loader, Noise_loader, Noise_Test_loader, Noise_Sample_loader, Noise_Triple_loader, All_loader, Test_loader, 1, 10

