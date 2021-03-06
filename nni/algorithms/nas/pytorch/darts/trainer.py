# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast
sys.path.append('../../../nni/') #nas/pytorch/

from nas.pytorch.trainer import Trainer
from nas.pytorch.utils_ import AverageMeterGroup
import numpy as np
from .mutator import DartsMutator

logger = logging.getLogger(__name__)


class SSLDartsTrainer(Trainer):
    """
    SSL-DARTS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset_train : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    dataset_valid : Dataset
        Dataset for testing.
    mutator : DartsMutator
        Use in case of customizing your own DartsMutator. By default will instantiate a DartsMutator.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    callbacks : list of Callback
        list of callbacks to trigger at events.
    arc_learning_rate : float
        Learning rate of architecture parameters.

    """
    def __init__(self, model, loss, metrics,
                 optimizer, num_epochs, dataset_train, dataset_valid,
                 mutator=None, batch_size=64, workers=4, device=None, log_frequency=None,
                 callbacks=None, arc_learning_rate=3.0E-4, temperature=0.07):
        super().__init__(model, mutator if mutator is not None else DartsMutator(model),
                         loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid,
                         batch_size, workers, device, log_frequency, callbacks)
        self.ctrl_optim = torch.optim.Adam(self.mutator.parameters(), arc_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
        self.temperature = temperature
        n_train = len(self.dataset_train)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=workers,
                                                        drop_last=True)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=workers,
                                                        drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=batch_size,
                                                       num_workers=workers,
                                                       drop_last=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        self.mutator.train()
        meters = AverageMeterGroup()
        loss_arc = []
        loss_w = []
        grad_norm_arc = []
        grad_norm_w = []
        
        for step, ((trn_X, _), (val_X, _)) in enumerate(zip(self.train_loader, self.valid_loader)):

            trn_X = torch.cat(trn_X, dim=0)
            val_X = torch.cat(val_X, dim=0)
            trn_X = trn_X.to(self.device)
            val_X = val_X.to(self.device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            loss_arc.append(self._backward(val_X).item())
            self.ctrl_optim.step()
            total_norm = 0
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
            grad_norm_arc.append(total_norm)
            
            # phase 2: child network step
            self.optimizer.zero_grad()
            logits, labels, loss = self._logits_and_loss(trn_X)
            loss_w.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.)  # gradient clipping
            self.optimizer.step()
            total_norm = 0
            for p in self.model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
            grad_norm_w.append(total_norm)
            
            metrics = self.metrics(logits, labels)
            metrics["loss"] = loss.item()
            meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                print("Epoch [{}/{}] Step [{}/{}]  {}".format(epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), meters))
        return np.mean(loss_arc), np.mean(loss_w), np.mean(grad_norm_arc), np.mean(grad_norm_w)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        self.mutator.eval()
        meters = AverageMeterGroup()
        with torch.no_grad():
            self.mutator.reset()
            for step, (X, _) in enumerate(self.test_loader):
                X = X.to(self.device)
                features = self.model(X)
                logits, labels = self.info_nce_loss(features)
                metrics = self.metrics(logits, labels)
                meters.update(metrics)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    print("Epoch [{}/{}] Step [{}/{}]  {}", epoch + 1,
                                self.num_epochs, step + 1, len(self.test_loader), meters)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        
        labels = labels[~mask].view(labels.shape[0], -1)
        
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
    
    def _logits_and_loss(self, X):
        self.mutator.reset()

        features = self.model(X)
        logits, labels = self.info_nce_loss(features)
        loss = self.loss(logits, labels)
        self._write_graph_status()
        return logits, labels, loss

    def _backward(self, val_X):
        """
        Simple backward with gradient descent
        """
        _, _, loss = self._logits_and_loss(val_X)
        loss.backward()
        return loss