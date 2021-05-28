# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10
from medmnist import ChestMNIST
from torchvision.transforms import transforms
from gaussian_blur import GaussianBlur
from view_generator import ContrastiveLearningViewGenerator
from longtailed_distr import reduce_classes_dbset_longtailed
from collections import defaultdict, deque
import itertools

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class CIFAR5000(CIFAR10):
    def __init__(self, path, transforms, train=True):
        super().__init__(path, train, download=True)
        self.transforms = transforms
        self.n_images_per_class = 1000
        self.n_classes = 10
        self.new2old_indices = self.create_idx_mapping()

    def create_idx_mapping(self):
        label2idx = defaultdict(lambda: deque(maxlen=self.n_images_per_class))
        for original_idx in range(super().__len__()):
            _, label = super().__getitem__(original_idx)
            label2idx[label].append(original_idx)

        old_idxs = set(itertools.chain(*label2idx.values()))
        new2old_indices = {}
        for new_idx, old_idx in enumerate(old_idxs):
            new2old_indices[new_idx] = old_idx

        return new2old_indices

    def __len__(self):
        return len(self.new2old_indices)

    def __getitem__(self, index):
        index = self.new2old_indices[index]
        im, label = super().__getitem__(index)
        return self.transforms(im), label
    
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
#         img = np.array(img)
#         h, w = img.shape[0], img.shape[1]
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)
#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)

#         mask[y1: y2, x1: x2] = 0.
#         mask = mask.astype(int)
# #         img = torch.from_numpy(img)
#         img = np.transpose(img, [2, 0, 1])#mask.unsqueeze(-1).expand_as(img)
#         img *= mask
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return transforms.ToPILImage()(img)


def get_dataset(cls, cutout_length=0):
#     MEAN = [0.49139968, 0.48215827, 0.44653124]
#     STD = [0.24703233, 0.24348505, 0.26158768]
    MEAN = [0.5]
    STD = [0.5]
    transf = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))
        cutout.append(transforms.ToTensor())

    train_transform = transforms.Compose(transf + normalize + cutout)
    valid_transform = transforms.Compose(normalize)

#     if cls == "cifar10":
#         dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
#     if cls == "cifar5000":
#         dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
#         dataset_train = torch.utils.data.Subset(dataset_train, np.arange(10000))
#     if cls == "nucifar10":
#         dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
#         # dataset_train = torch.utils.data.Subset(dataset_train, np.arange(10000))
#         dataset_train.data = dataset_train.data[:10000]
#         dataset_train.targets = dataset_train.targets[:10000]
#         dataset_train.data, dataset_train.targets, cls_dist = reduce_classes_dbset_longtailed(dataset_train, lt_factor=0.8)
#     dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    dataset_train = ChestMNIST(root="./data", split='train', transform=train_transform, download=True)
#     dataset_train.data = dataset_train.data[:10000]
#     dataset_train.targets = dataset_train.targets[:10000]
#     dataset_train.data, dataset_train.targets, cls_dist = reduce_classes_dbset_longtailed(dataset_train, lt_factor=0.8)
#     dataset_train.targets = np.expand_dims((dataset_train.targets.sum(1) != 0).astype(np.int), -1)  
    dataset_valid = ChestMNIST(root="./data", split='test', transform=valid_transform, download=True)
#     dataset_valid.targets = np.expand_dims((dataset_valid.targets.sum(1) != 0).astype(np.int), -1)
    
    return dataset_train, dataset_valid#, cls_dist

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, cutout_length=0):
#         MEAN = [0.49139968, 0.48215827, 0.44653124]
#         STD = [0.24703233, 0.24348505, 0.26158768]
        MEAN = [0.5]
        STD = [0.5]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
#         dataset_train = CIFAR10(self.root_folder, train=True,
#                   transform=transforms.Compose(normalize+[transforms.ToPILImage(),#Cutout(cutout_length), 
#         ContrastiveLearningViewGenerator(
#                   self.get_simclr_pipeline_transform(32),
#                   2)]),
#                   download=True)
#         dataset_train.data, dataset_train.targets, cls_dist = reduce_classes_dbset_longtailed(dataset_train, lt_factor=0.8)
#         dataset_valid = CIFAR10(self.root_folder, train=False,
#                                   transform=transforms.Compose(normalize+[transforms.ToPILImage(), ContrastiveLearningViewGenerator(
#                                   self.get_simclr_pipeline_transform(32),
#                                   2)]),
#                                   download=True)
        dataset_train = ChestMNIST(root=self.root_folder, split='train',
                  transform=transforms.Compose(normalize+[transforms.ToPILImage(),#Cutout(cutout_length), 
			ContrastiveLearningViewGenerator(
                  self.get_simclr_pipeline_transform(28),
                  2)]),
                  download=True)
#         dataset_train.targets = (dataset_train.targets.sum(1) != 0).astype(np.int)
        
#         dataset_train.data, dataset_train.targets, cls_dist = reduce_classes_dbset_longtailed(dataset_train, lt_factor=0.8)
        dataset_valid = ChestMNIST(root=self.root_folder, split='test',
                                  transform=transforms.Compose(normalize+[transforms.ToPILImage(), ContrastiveLearningViewGenerator(
                                  self.get_simclr_pipeline_transform(28),
                                  2)]),
                                  download=True)
#         dataset_valid.targets = (dataset_valid.targets.sum(1) != 0).astype(np.int)
    
        return dataset_train, dataset_valid#, cls_dist
