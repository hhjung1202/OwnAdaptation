import os
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import numpy as np

import torch
import torchvision.datasets as datasets


class CIFAR10NoisyLabels(datasets.CIFAR10):
    """CIFAR10 Dataset with noisy labels.

    Args:
        noise_type (string): Noise type (default: 'symmetric').
            The value is either 'symmetric' or 'asymmetric'.
        noise_rate (float): Probability of label corruption (default: 0.0).
        seed (int): Random seed (default: 12345).
        
    This is a subclass of the `CIFAR10` Dataset.
    """

    def __init__(self,
                 noise_type='symmetric',
                 noise_rate=0.0,
                 seed=12345,
                 **kwargs):
        super(CIFAR10NoisyLabels, self).__init__(**kwargs)
        self.seed = seed
        self.num_classes = 10
        self.flip_pairs = np.asarray([[9, 1], [2, 0], [4, 7], [3, 5], [5, 3]])

        if noise_rate > 0:
            if noise_type == 'symmetric':
                self.symmetric_noise(noise_rate)
            elif noise_type == 'asymmetric':
                self.asymmetric_noise(noise_rate)
            else:
                raise ValueError(
                    'expected noise_type is either symmetric or asymmetric '
                    '(got {})'.format(noise_type))

    def symmetric_noise(self, noise_rate):
        """Insert symmetric noise.

        For all classes, ground truth labels are replaced with uniform random
        classes.
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        mask = np.random.rand(len(targets)) <= noise_rate
        rnd_targets = np.random.choice(self.num_classes, mask.sum())
        targets[mask] = rnd_targets
        targets = [int(target) for target in targets]
        self.targets = targets

    def asymmetric_noise(self, noise_rate):
        """Insert asymmetric noise.

        Ground truth labels are flipped by mimicking real mistakes between
        similar classes. Following `Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach`_, 
        ground truth labels are replaced with
        
        * truck -> automobile,
        * bird -> airplane,
        * deer -> horse
        * cat -> dog
        * dog -> cat

        .. _Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
            https://arxiv.org/abs/1609.03683
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        for i, target in enumerate(targets):
            if target in self.flip_pairs[:, 0]:
                if np.random.uniform(0, 1) <= noise_rate:
                    idx = int(np.where(self.flip_pairs[:, 0] == target)[0])
                    targets[i] = self.flip_pairs[idx, 1]
        targets = [int(x) for x in targets]
        self.targets = targets

    def T(self, noise_type, noise_rate):
        if noise_type == 'symmetric':
            T = (torch.eye(self.num_classes) * (1 - noise_rate) +
                 (torch.ones([self.num_classes, self.num_classes]) /
                  self.num_classes * noise_rate))
        elif noise_type == 'asymmetric':
            T = torch.eye(self.num_classes)
            for i, j in self.flip_pairs:
                T[i, i] = 1 - noise_rate
                T[i, j] = noise_rate
        return T


class CIFAR100NoisyLabels(datasets.CIFAR100):
    """CIFAR100 Dataset with noisy labels.

    Args:
        noise_type (string): Noise type (default: 'symmetric').
            The value is either 'symmetric' or 'asymmetric'.
        noise_rate (float): Probability of label corruption (default: 0.0).
        seed (int): Random seed (default: 12345).

    This is a subclass of the `CIFAR100` Dataset.
    """

    def __init__(self,
                 noise_type='synmetric',
                 noise_rate=0.0,
                 seed=12345,
                 **kwargs):
        super(CIFAR100NoisyLabels, self).__init__(**kwargs)
        self.seed = seed
        self.num_classes = 100
        self.num_superclasses = 20

        if noise_rate > 0:
            if noise_type == 'symmetric':
                self.symmetric_noise(noise_rate)
            elif noise_type == 'asymmetric':
                self.asymmetric_noise(noise_rate)
            else:
                raise ValueError(
                    'expected noise_type is either symmetric or asymmetric '
                    '(got {})'.format(noise_type))

    def symmetric_noise(self, noise_rate):
        """Symmetric noise in CIFAR100.

        For all classes, ground truth labels are replaced with uniform random
        classes.
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        mask = np.random.rand(len(targets)) <= noise_rate
        rnd_targets = np.random.choice(self.num_classes, mask.sum())
        targets[mask] = rnd_targets
        targets = [int(x) for x in targets]
        self.targets = targets

    def asymmetric_noise(self, noise_rate):
        """Insert asymmetric noise.

        Ground truth labels are flipped by mimicking real mistakes between
        similar classes. Following `Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach`_, 
        ground truth labels are flipped into the next class circularly within
        the same superclasses

        .. _Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
            https://arxiv.org/abs/1609.03683
        """
        np.random.seed(self.seed)
        targets = np.array(self.targets)
        Tdata = self.T('asymmetric', noise_rate).numpy().astype(np.float64)
        Tdata = Tdata / np.sum(Tdata, axis=1)[:, None]
        for i, target in enumerate(targets):
            one_hot = np.random.multinomial(1, Tdata[target, :], 1)[0]
            targets[i] = np.where(one_hot == 1)[0]
        targets = [int(x) for x in targets]
        self.targets = targets

    def _load_coarse_targets(self):
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        coarse_targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                coarse_targets.extend(entry['coarse_labels'])

        return coarse_targets

    def T(self, noise_type, noise_rate):
        if noise_type == 'symmetric':
            T = (torch.eye(self.num_classes) * (1 - noise_rate) +
                 (torch.ones([self.num_classes, self.num_classes]) /
                  self.num_classes * noise_rate))
        elif noise_type == 'asymmetric':
            num_classes = self.num_classes
            num_superclasses = self.num_superclasses
            num_subclasses = num_classes // num_superclasses

            targets = np.array(self.targets)
            coarse_targets = np.asarray(self._load_coarse_targets())

            T = torch.eye(num_classes) * (1 - noise_rate)
            for i in range(num_superclasses):
                subclass_targets = np.unique(targets[coarse_targets == i])
                clean = subclass_targets
                noisy = np.concatenate([clean[1:], clean[:1]])
                for j in range(num_subclasses):
                    T[clean[j], noisy[j]] =  
        return T



from torchvision.datasets import CIFAR10
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class _Symetric_Noise:
    def __init__(self, probablity=.1, sym=True, seed=1234):
        self.prob = probablity
        self.sym = sym
        self.seed=seed

    def __call__(self, x):
        torch.manual_seed(self.seed)
        item = torch.randint(0, 9, size=[len(x)])
        random = torch.rand(len(x))
        x = torch.as_tensor(x)
        item = torch.where(self.prob <= random, x, item)

        # Symmetric
        if self.sym:
            return np.vstack([item, x]).swapaxes(0, 1)

        # ASymmetric
        else:
            for i in range(len(x)):
                if self.prob >= random[i]:
                    if x[i] == 9:
                        item[i] = 1
                        # return torch.tensor([1]), x
                        # bird -> airplane
                    elif x[i] == 2:
                        item[i] = 0
                        # return torch.tensor([0]), x
                        # cat -> dog
                    elif x[i] == 3:
                        item[i] = 5
                        # return torch.tensor([5]), x
                        # dog -> cat
                    elif x[i] == 5:
                        item[i] = 3
                        # return torch.tensor([3]), x
                        # deer -> horse
                    elif x[i] == 4:
                        item[i] = 7
                        # return torch.tensor([7]), x
            return np.vstack([item, x]).swapaxes(0, 1)


class Noise_CIFAR10:
    def __init__(self, root, train=True, transforms=None, down=True, noise_rate=.1, sym=True, seed=1234):
        self.seed = seed
        self.dataset = CIFAR10(root, train=train, transform=transforms, download=down)#, target_transform=_Symetric_Noise(noise_rate, sym, seed), )
        self.dataset.targets = _Symetric_Noise(noise_rate, sym, seed)(self.dataset.targets)
        # print((self.dataset.targets[:, 0] == self.dataset.targets[:, 1]).sum())
        # print((self.dataset.targets[:, 1] == 9).sum())
        # print((self.dataset.targets[self.dataset.targets[:, 1] == 9, 0] != 9).sum() + (self.dataset.targets[self.dataset.targets[:, 1] == 9, 0] != 1).sum())

    def get_loader(self, **kwargs):
        return DataLoader(self.dataset, **kwargs)