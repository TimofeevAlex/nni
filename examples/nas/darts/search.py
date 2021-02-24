# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import json
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from model import CNN
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from utils import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--unrolled", default=False, action="store_true")
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument("--supervised", default=False, action="store_true")
    args = parser.parse_args()
    
    n_classes = 10 if args.supervised else 128
    model = CNN(32, 3, args.channels, n_classes, args.layers)
    if not args.supervised:
        model.linear = nn.Sequential(nn.Linear(model.linear.in_features, model.linear.in_features), nn.ReLU(), model.linear)
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)
    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    if args.supervised:
        from nni.algorithms.nas.pytorch.darts import DartsTrainer
        dataset = datasets.ContrastiveLearningDataset('./data')
        dataset_train = dataset.get_dataset('cifar10', 1)
        _, dataset_valid = datasets.get_dataset("cifar10")
#         dataset_train, dataset_valid = datasets.get_dataset("cifar10")
        trainer = DartsTrainer(model,
                               loss=criterion,
                               metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                               optimizer=optim,
                               num_epochs=args.epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               batch_size=args.batch_size,
                               log_frequency=args.log_frequency,
                               unrolled=args.unrolled,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
        if args.visualization:
            trainer.enable_visualization()

        trainer.train()
        trainer.validate()
        trainer.export('checkpoint.json')

    else:
        sys.path.append('../../../nni/algorithms/nas/pytorch/')
        from darts import SSLDartsTrainer
        dataset = datasets.ContrastiveLearningDataset('./data')
        dataset_train, dataset_valid = dataset.get_dataset()
#         _, dataset_valid = datasets.get_dataset("cifar10")

        trainer = SSLDartsTrainer(model,
                       loss=criterion,
                       metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                       optimizer=optim,
                       num_epochs=args.epochs,
                       dataset_train=dataset_train,
                       dataset_valid=dataset_valid,
                       batch_size=args.batch_size,
                       log_frequency=args.log_frequency,
                       device=device,
                       callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")])
        trainer.train(validate=False)
#         trainer.validate()
        trainer.export('ssl_checkpoint.json')
