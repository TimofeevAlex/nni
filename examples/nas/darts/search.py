# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import json
import logging
import time
from datetime import datetime 
from argparse import ArgumentParser

import torch
import torch.nn as nn

import datasets
from model import CNN
sys.path.append('../../../nni/nas/pytorch/')
from callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from utils import accuracy

import matplotlib.pyplot as plt 
sys.path.append('../../../nni/algorithms/nas/pytorch/')
from darts import SSLDartsTrainer
from darts import DartsMutator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=8, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--channels", default=16, type=int)
    parser.add_argument("--visualization", default=False, action="store_true")
    parser.add_argument("--save-to", default='ssl_checkpoints.json', type=str)
    parser.add_argument("--temperature", default=0.07, type=float)
    
    args = parser.parse_args()
    
    model = CNN(32, 3, args.channels, 128, args.layers)
    model.linear = nn.Sequential(nn.Linear(model.linear.in_features, model.linear.in_features), nn.ReLU(), model.linear)
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)
    optim = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)

    dataset = datasets.ContrastiveLearningDataset('./data')
    dataset_train, dataset_valid = dataset.get_dataset()

    trainer = SSLDartsTrainer(model,
                   loss=criterion,
                   metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                   optimizer=optim,
                   num_epochs=args.epochs,
                   dataset_train=dataset_train,
                   dataset_valid=dataset_valid,
                   batch_size=args.batch_size,
                   log_frequency=args.log_frequency,
                   mutator=DartsMutator(model),
                   device=device,
                   callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("./checkpoints")],
                   temperature=args.temperature)
    trainer.enable_visualization()
    loss_arc, loss_w, grad_norm_arc, grad_norm_w = trainer.train(validate=False)
    trainer.export(args.save_to)
    
    try:  
        os.mkdir('plots')  
    except OSError as error:  
        print(error)
        
    timenow = str(datetime.now()).replace('-', '').replace(' ', '').replace(':', '').replace('.', '')
       
    fig, ax = plt.subplots()
    ax.plot(loss_arc, label='Architecture loss')
    ax.plot(loss_w, label='Weights loss')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
#     ax.set_title('Architecture loss')
    plt.savefig('plots/search_arch_loss_'+ timenow + '.png')
    
    fig, ax = plt.subplots()
    ax.plot(grad_norm_arc, label='Architecture grad norm')
    ax.plot(grad_norm_w, label='Weights grad norm')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Norm')
#     ax.set_title('Architecture loss')
    plt.savefig('plots/search_arch_grad_'+ timenow + '.png')

