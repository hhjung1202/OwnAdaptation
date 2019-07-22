from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, label_path, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.label_path, 'r')]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            id_ = split[1]

            self.train_dataset.append([filename, id_])

            if i <= 2000:
                self.test_dataset.append([filename, id_])
            else:
                self.train_dataset.append([filename, id_])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        w, h = image.size
        cropImage = torch.Tensor(image.crop((w/2-65, h/2 - 40, w/2 + 65 , h/2 + 90)))
        print(self.transform(cropImage))
        return self.transform(cropImage), torch.FloatTensor(int(label))

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def CelebA_loader(image_size=32, batch_size=16, num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Grayscale(1))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    train_dataset = CelebA( "/database/data/celeba/images"
                    , "/database/data/celeba/largest_identity_CelebA_4000.txt"
                    , transform
                    , 'train')

    test_dataset = CelebA( "/database/data/celeba/images"
                    , "/database/data/celeba/largest_identity_CelebA_4000.txt"
                    , transform
                    , 'test')

    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=(True),
                                  num_workers=num_workers)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=(False),
                                  num_workers=num_workers)

    return train_loader, test_loader