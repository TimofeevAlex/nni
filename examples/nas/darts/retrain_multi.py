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

import datasets_multi
import utils
from utils import accuracy
from model import CNN
from focal_loss import FocalLoss

sys.path.append('../../../nni/nas/pytorch/')

from utils_ import AverageMeter
from fixed import apply_fixed_architecture
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from metrics import getAUC, getACC

np.random.seed(42)

logger = logging.getLogger('nni')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def train(config, train_loader, model, optimizer, criterion, epoch):#, cls_dist):
    top1 = AverageMeter("top1")
#     top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)

    model.train()
    losses_ = []
    grad_norm_w = []
    for step, (x, y) in enumerate(train_loader):

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)
      
        optimizer.zero_grad()
        logits = model(x)
#         logits += torch.log(1. / cls_dist.to(device).float())
        y = y.to(torch.float32)#.squeeze().long()
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

        accuracy = utils.accuracy(logits, y)#topk=(1, 5)
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
#         top5.update(accuracy["acc5"], bs)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
#         writer.add_scalar("acc5/train", accuracy["acc5"], global_step=cur_step)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            print("Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%})".format(# , {top5.avg:.1%})
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1))#, top5=top5))

        cur_step += 1

    print("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
    return np.mean(losses_), np.mean(grad_norm_w) 


def validate(config, valid_loader, model, criterion, epoch, cur_step):#, cls_dist):
    top1 = AverageMeter("top1")
#     top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()
    y_true = torch.tensor([])
    y_pred1 = torch.tensor([])
    y_pred2 = torch.tensor([])
    
    losses_val = []
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            y = y.to(torch.float32)#.squeeze().long()
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)
            losses_val.append(loss.item())
            accuracy = utils.accuracy(logits, y)#topk=(1, 5)
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            # top5.update(accuracy["acc5"], bs)
            
#             y = y.float().resize_(len(y), 1)
            y_true = torch.cat((y_true, y.detach().cpu()), 0)
            y_pred1 = torch.cat((y_pred1, torch.argmax(logits, 1).detach().cpu()), 0)
            y_pred2 = torch.cat((y_pred2, nn.Sigmoid()(logits).detach().cpu()))

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                print("Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%})".format(#, {top5.avg:.1%}
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1))#, top5=top5))

    writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
    writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
#     writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)
    print("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))
    y_true = y_true.numpy()
    y_pred1 = y_pred1.numpy()
    y_pred2 = y_pred2.numpy()
    auc = getAUC(y_true, y_pred2, 'multi-label, binary-class') #
    acc = getACC(y_true, y_pred2, 'multi-label, binary-class') #
    print("Valid: [{:3d}/{}] AUC: {:.4%} ACC: {:.4%}".format(epoch + 1, config.epochs, auc, acc))
    # Confusion matrix
#     conf_mat = confusion_matrix(y_true, y_pred1)
#     class_labels= ['airplanes', 'cars', 'birds', 'cats', 'deer',\
#                    'dogs', 'frogs', 'horses', 'ships', 'trucks']
    
#     class_labels = ["Negative", "Positive"]

#     conf_mat = pd.DataFrame(conf_mat, columns=class_labels, index=class_labels)
#     mask = np.ones_like(conf_mat)
#     # mask[np.triu_indices_from(mask)] = False
#     plt.figure(figsize=(10, 10))
#     ax = sns.heatmap(conf_mat, cmap='coolwarm', annot=True)#mask=mask
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right')
#     ax.set_title('Confusion matrix heatmap')
#     plt.tight_layout()
#     if config.no_pretrained:
#         if args.not_reinit != 'None':
#             plt.savefig('plots/nossl_confusion_matrix '+config.dataset+'.png')
#         else:
#             plt.savefig('plots/new_weights_nossl_confusion_matrix '+config.dataset+'.png')
#     else:
#         plt.savefig('plots/confusion_matrix '+config.dataset+'.png')
    
    
    
    return top1.avg, np.mean(losses_val)


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--n-nodes", default=4, type=int)
    parser.add_argument("--stem-multiplier", default=3, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--pretrained", default="./checkpoints/epoch_0.json")
    parser.add_argument("--no-pretrained", default=False, type=bool)
    parser.add_argument("--drop-path-prob", default=0.2, type=float)
    parser.add_argument("--channels", default=36, type=int)
    parser.add_argument("--dataset", default='cifar5000')
    parser.add_argument("--arc-checkpoint", default="./checkpoints/epoch_0.json")
    parser.add_argument("--not-reinit", default='supernet_models/supernet_epoch_30')

    args = parser.parse_args()
    
    if args.no_pretrained:
        model = CNN(28, 1, args.channels, 128, args.layers, auxiliary=False, n_nodes=args.n_nodes, stem_multiplier=args.stem_multiplier)
        model.linear = nn.Sequential(nn.Linear(model.linear.in_features, model.linear.in_features), nn.ReLU(), model.linear)
        if args.not_reinit != 'None':
            model.load_state_dict(torch.load(args.not_reinit))
        apply_fixed_architecture(model, args.arc_checkpoint)
    else:
        model = torch.load(args.pretrained)
    
    model = nn.Sequential(model, nn.ReLU(), nn.Linear(128, 14)) #dropout

    dataset_train, dataset_valid = datasets_multi.get_dataset(args.dataset, cutout_length=0)# FIX TO 10%
    criterion = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()#nn.BCELoss()#FocalLoss(gamma=2.)##Add alphas

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4) #0.025
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.5)    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6) #reduce on plateau
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, threshold=0.001) 
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
    main_model = [module for i, module in enumerate(model.modules()) if i != 0][0]
    for epoch in range(args.epochs):
        # training
        drop_prob = args.drop_path_prob * epoch / args.epochs
        main_model.drop_path_prob(drop_prob)
        loss_ep, grad_norm_w_ep = train(args, train_loader, model, optimizer, criterion, epoch)#, cls_dist)
        losses.append(loss_ep)
        grad_norm_w.append(grad_norm_w_ep)
        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1, loss_val_ep = validate(args, valid_loader, model, criterion, epoch, cur_step)#, cls_dist)
        losses_val.append(loss_val_ep)
        best_top1 = max(best_top1, top1)

        lr_scheduler.step()
        torch.save(model, os.path.join(models_dir, 'model'+'_'+str(epoch)+'_'+args.dataset+'.pt'))
        
        if epoch % 5 == 0:
            fig, ax = plt.subplots()
            ax.plot(losses, label='Train loss')
            ax.plot(losses_val, label='Validation loss')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            if args.no_pretrained:
                if args.not_reinit != 'None':
                    plt.savefig('plots/nossl_training_loss_epoch_'+ str(epoch) + '_' + args.dataset + '_'+ timenow + '.png')
                else:
                    plt.savefig('plots/new_weights_nossl_training_loss_epoch_'+ str(epoch) + '_' + args.dataset + '_'+ timenow + '.png')
            else:
                plt.savefig('plots/supervised_training_loss_epoch_'+ str(epoch) + '_' + args.dataset + '_'+ timenow + '.png')

            fig, ax = plt.subplots()
            ax.plot(grad_norm_w, label='Norm')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Norm')
            if args.no_pretrained:
                if args.not_reinit != 'None':
                    plt.savefig('plots/nossl_training_grad_norm_epoch_'+ str(epoch) + '_' + args.dataset + '_'+ timenow + '.png')
                else:
                    plt.savefig('plots/new_weights_nossl_training_grad_norm_epoch_'+ str(epoch) + '_' + args.dataset + '_'+ timenow + '.png')
            else:
                plt.savefig('plots/supervised_training_grad_norm_epoch_'+ str(epoch) + '_' + args.dataset + '_'+timenow + '.png')

    print("Final best Prec@1 = {:.4%}".format(best_top1))
