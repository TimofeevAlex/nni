# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import logging
import time
from datetime import datetime 
from argparse import ArgumentParser

import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import datasets
import utils
from utils import accuracy
from model import CNN
sys.path.append('../../../nni/nas/pytorch/')
from fixed import apply_fixed_architecture
from utils_ import AverageMeter
import matplotlib.pyplot as plt 

logger = logging.getLogger('nni')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def info_nce_loss(features, config):
        labels = torch.cat([torch.arange(config.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        
        labels = labels[~mask].view(labels.shape[0], -1)
        
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / config.temperature
        return logits, labels

def ssl_train(config, train_loader, model, optimizer, criterion, epoch):
    model.train()
    losses = AverageMeter("losses")
    losses_ = []
    grad_norm_w = []
    for step, (trn_X, _) in enumerate(train_loader):

        trn_X = torch.cat(trn_X, dim=0)
        trn_X = trn_X.to(device)

        optimizer.zero_grad()
        features = model(trn_X)
        logits, labels = info_nce_loss(features, config)
        loss = criterion(logits, labels)
        losses_.append(loss.item())
        loss.backward()   
        nn.utils.clip_grad_norm_(model.parameters(), 5.)  # gradient clipping
        optimizer.step()
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        grad_norm_w.append(total_norm)

        losses.update(loss.item(), config.batch_size)
        if config.log_frequency is not None and step % config.log_frequency == 0:
            print("Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                        epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses))    

    return np.mean(losses_), np.mean(grad_norm_w)

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--arc-checkpoint", default="./checkpoints/epoch_0.json")
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--channels", default=36, type=int)
    parser.add_argument("--keep-training", default=None, type=str)

    args = parser.parse_args()
    
    if args.keep_training != None:
        model = torch.load(args.keep_training)
    else:
        model = CNN(32, 3, args.channels, 128, args.layers, auxiliary=False)
        model.linear = nn.Sequential(nn.Linear(model.linear.in_features, model.linear.in_features), nn.ReLU(), model.linear)
   
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)
    
    dataset = datasets.ContrastiveLearningDataset('./data')
    dataset_train, _ = dataset.get_dataset()
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=args.batch_size,
                                                num_workers=args.workers,
                                                drop_last=True)

    try:  
        os.mkdir('models')  
    except OSError as error:  
        print(error)   
    timenow = str(datetime.now()).replace('-', '').replace(' ', '').replace(':', '').replace('.', '')
    models_dir = os.path.join('models', timenow)
    losses = []
    grad_norm_w = []
    os.mkdir(models_dir)
    for epoch in range(args.epochs):
        loss_ep, grad_norm_w_ep = ssl_train(args, train_loader, model, optimizer, criterion, epoch)
        losses.append(loss_ep)
        grad_norm_w.append(grad_norm_w_ep)
        torch.save(model, os.path.join(models_dir, 'model'+'_'+str(epoch)))

    torch.save(model, os.path.join(models_dir, 'model_final'))
    
    try:  
        os.mkdir('plots')  
    except OSError as error:  
        print(error)
        
    fig, ax = plt.subplots()
    ax.plot(losses, label='Loss')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
#     ax.set_title('Architecture loss')
    plt.savefig('plots/ssl_training_loss_'+ timenow + '.png')
    
    fig, ax = plt.subplots()
    ax.plot(grad_norm_w, label='Norm')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Norm')
    ax.set_ylabel('Loss')
#     ax.set_title('Architecture loss')
    plt.savefig('plots/ssl_training_grad_norm_'+ timenow + '.png')
