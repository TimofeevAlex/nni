# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import numpy as np
import logging
import time
from datetime import datetime 

from argparse import ArgumentParser
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import datasets
import utils
from utils import accuracy
from model import CNN

sys.path.append('../../../nni/nas/pytorch/')

from utils_ import AverageMeter
from fixed import apply_fixed_architecture
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

logger = logging.getLogger('nni')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

def train(config, train_loader, model, optimizer, criterion, epoch):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step=cur_step)

    model.train()
    losses_ = []
    grad_norm_w = []
    for step, (x, y) in enumerate(train_loader):

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        losses_.append(loss.item())
        loss.backward()
        # gradient clipping
#         nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        grad_norm_w.append(total_norm)

        accuracy = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
        top5.update(accuracy["acc5"], bs)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
        writer.add_scalar("acc5/train", accuracy["acc5"], global_step=cur_step)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            print("Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
    return np.mean(losses_), np.mean(grad_norm_w) 


def validate(config, valid_loader, model, criterion, epoch, cur_step):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    losses_val = []
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            y_true = np.append(y_true, np.array(y))
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            logits = model(X)
            y_pred = np.append(y_pred, np.argmax(logits.cpu().numpy(), axis=1))
        
            loss = criterion(logits, y)
            losses_val.append(loss.item())
            accuracy = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            top5.update(accuracy["acc5"], bs)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                print("Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
    writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
    writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
    
    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    class_labels= ['airplanes', 'cars', 'birds', 'cats', 'deer',\
                   'dogs', 'frogs', 'horses', 'ships', 'trucks']
    conf_mat = pd.DataFrame(conf_mat, columns=class_labels, index=class_labels)
    mask = np.ones_like(conf_mat)
    mask[np.triu_indices_from(mask)] = False
    ax = sns.heatmap(conf_mat, mask=mask, cmap='coolwarm', annot=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right')
    ax.set_title('Confusion matrix heatmap')
    plt.savefig('plots/confusion_matrix'+config.dataset+'.png')
    return top1.avg, np.mean(losses_val)


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--pretrained", default="./checkpoints/epoch_0.json")

    parser.add_argument("--channels", default=36, type=int)
    parser.add_argument("--dataset", default='cifar5000')
    args = parser.parse_args()
    

    model = torch.load(args.pretrained)

    model = nn.Sequential(model, nn.ReLU(), nn.Linear(128, 10))
    dataset_train, dataset_valid = datasets.get_dataset(args.dataset)# FIX TO 10%
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    best_top1 = 0.
    try:  
        os.mkdir('models_super')  
    except OSError as error:  
        print(error)     
    try:  
        os.mkdir('plots')  
    except OSError as error:  
        print(error)
        
    timenow = str(datetime.now()).replace('-', '').replace(' ', '').replace(':', '').replace('.', '')
    models_dir = os.path.join('models_super', timenow)
    os.mkdir(models_dir)
    losses = []
    losses_val = []
    grad_norm_w = []
    for epoch in range(args.epochs):
        # training
        loss_ep, grad_norm_w_ep = train(args, train_loader, model, optimizer, criterion, epoch)
        losses.append(loss_ep)
        grad_norm_w.append(grad_norm_w_ep)
        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1, loss_val_ep = validate(args, valid_loader, model, criterion, epoch, cur_step)
        losses_val.append(loss_val_ep)
        best_top1 = max(best_top1, top1)

        lr_scheduler.step()
        torch.save(model, os.path.join(models_dir, 'model'+'_'+str(epoch)+'_'+config.dataset+'.pt'))
        
        if epoch % 5 == 0:
            fig, ax = plt.subplots()
            ax.plot(losses, label='Train loss')
            ax.plot(losses_val, label='Validation loss')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            plt.savefig('plots/supervised_training_loss_epoch_'+ str(epoch) + '_' + config.dataset + '_'+ timenow + '.png')

            fig, ax = plt.subplots()
            ax.plot(grad_norm_w, label='Norm')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Norm')
            plt.savefig('plots/supervised_training_grad_norm_epoch_'+ str(epoch) + '_' + config.dataset + '_'+timenow + '.png')

    print("Final best Prec@1 = {:.4%}".format(best_top1))
