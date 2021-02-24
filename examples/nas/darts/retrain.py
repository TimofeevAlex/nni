# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import datasets
import utils
from utils import accuracy
from model import CNN
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter

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
    for step, (trn_X, _) in enumerate(train_loader):
        trn_X = torch.cat(trn_X, dim=0)
        trn_X = trn_X.to(device)


        optimizer.zero_grad()
        features = model(trn_X)
        logits, labels = info_nce_loss(features, config)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)  # gradient clipping
        optimizer.step()

        losses.update(loss.item(), config.batch_size)
        if config.log_frequency is not None and step % config.log_frequency == 0:
            logger.info("Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                        epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses))

def train(config, train_loader, model, optimizer, criterion, epoch):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step=cur_step)

    model.train()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        accuracy = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
        top5.update(accuracy["acc5"], bs)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
        writer.add_scalar("acc5/train", accuracy["acc5"], global_step=cur_step)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))


def validate(config, valid_loader, model, criterion, epoch, cur_step):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)

            accuracy = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            top5.update(accuracy["acc5"], bs)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
    writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
    writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    return top1.avg


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--drop-path-prob", default=0.2, type=float)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--grad-clip", default=5., type=float)
    parser.add_argument("--arc-checkpoint", default="./checkpoints/epoch_0.json")
    parser.add_argument("--temperature", default=0.07, type=float)

    args = parser.parse_args()
    

    model = CNN(32, args.layers, 36, 128, args.layers, auxiliary=False)
    model.linear = nn.Sequential(nn.Linear(model.linear.in_features, model.linear.in_features), nn.ReLU(), model.linear)
    apply_fixed_architecture(model, args.arc_checkpoint)
   
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
    for epoch in range(args.epochs):
        ssl_train(args, train_loader, model, optimizer, criterion, epoch)
    
    # SUPERVISED
    for param in model.parameters():
        param.requires_grad = False
    model = nn.Sequential(model, nn.ReLU(), nn.Linear(128, 10))
    dataset_train, dataset_valid = datasets.get_dataset("cifar10")# FIX TO 10%
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)
    if args.supervised:
        train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.workers,
                                           pin_memory=True)
    else:
        n_train = len(dataset_train)
        split = n_train #// 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        train_loader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=args.batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=args.workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    best_top1 = 0.
    for epoch in range(args.epochs):
#         drop_prob = args.drop_path_prob * epoch / args.epochs
#         model.drop_path_prob(drop_prob)

        # training
        train(args, train_loader, model, optimizer, criterion, epoch)

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(args, valid_loader, model, criterion, epoch, cur_step)
        best_top1 = max(best_top1, top1)

        lr_scheduler.step()

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
