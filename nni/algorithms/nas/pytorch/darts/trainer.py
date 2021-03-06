# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import copy
import logging
import scipy.sparse.linalg as linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast
sys.path.append('../../../nni/') #nas/pytorch/

from nas.pytorch.trainer import Trainer
from nas.pytorch.utils_ import AverageMeterGroup
import numpy as np
from .mutator import DartsMutator
from .jvp import JacobianVectorProduct
from .extragradient import ExtraAdam
from torch import autograd
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
        rand_step = np.random.randint(0, 10)
        for step, ((trn_X, _), (val_X, _)) in enumerate(zip(self.train_loader, self.valid_loader)):

            trn_X = torch.cat(trn_X, dim=0)
            val_X = torch.cat(val_X, dim=0)
            trn_X = trn_X.to(self.device)
            val_X = val_X.to(self.device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            loss_alpha = self._backward(val_X)
            loss_arc.append(loss_alpha.item())
            self.ctrl_optim.step()
#             if self.ctrl_optim.extrapolated:
#                 self.ctrl_optim.extrapolation()
#                 self.ctrl_optim.extrapolated = False
#             else:
#                 self.ctrl_optim.step()
#                 self.ctrl_optim.extrapolated = True
            
            total_norm = 0
            grads = []
            for p in self.mutator.parameters():
                grads.append(p.grad.data)
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
            grad_norm_arc.append(total_norm)
            # phase 2: child network step
            self.optimizer.zero_grad()
            loss = self._logits_and_loss(trn_X)
            loss_w.append(loss.item())
            loss.backward()
            self.optimizer.step()
#             if self.optimizer.extrapolated:
#                 self.optimizer.extrapolation()
#                 self.optimizer.extrapolated = False
#             else:
#                 self.optimizer.step()
#                 self.optimizer.extrapolated = True

            total_norm = 0
            for p in self.model.parameters():
                grads.append(p.grad.data)
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
            grad_norm_w.append(total_norm)
            
#             metrics = self.metrics(logits, labels)
#             metrics["loss"] = loss.item()
#             meters.update(metrics)
            if self.log_frequency is not None and step % self.log_frequency == 0:
                print("Epoch [{}/{}] Step [{}/{}]  {}".format(epoch + 1,
                            self.num_epochs, step + 1, len(self.train_loader), loss.item()))
                
#                 if (step == rand_step) and (epoch % 5 == 0):
#                     _, _, loss_x = self._logits_and_loss(val_X)
#                     gradx = autograd.grad(loss_x, self.mutator.parameters(), create_graph=True, allow_unused=True)
#                     _, _, loss_y = self._logits_and_loss(trn_X)
#                     grady = autograd.grad(loss_y, self.model.parameters(), create_graph=True, allow_unused=True)
#                     J = JacobianVectorProduct(list(gradx) + list(grady), params, force_numpy=True)
#                     dis_eigs = linalg.eigs(J, k=100, which='LI')[0]
#                     np.save('eigenvals/max_' + str(epoch), dis_eigs.imag.max())
#                     np.save('eigenvals/min_' + str(epoch), dis_eigs.imag.min())
        
        return np.mean(loss_arc), np.mean(loss_w), np.mean(grad_norm_arc), np.mean(grad_norm_w)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        self.mutator.eval()
        losses = []
        with torch.no_grad():
            self.mutator.reset()
            for step, (X, y) in enumerate(self.test_loader):
                X = torch.cat(X, dim=0)
                X = X.to(self.device)
                features = self.model(X)
#                 logits, labels = self.info_nce_loss(features)
                loss = self.info_nce_loss(features)#self.loss(logits, labels)
                losses.append(loss.item())
                if step == 0:
                    Xs = features[:self.batch_size]
                    ys = y
                Xs = torch.cat([Xs, features[:self.batch_size]])
                ys = torch.cat([ys, y])
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    print("Epoch [{}/{}] Step [{}/{}]  {}".format(epoch + 1,
                                self.num_epochs, step + 1, len(self.test_loader), loss))
        return np.mean(losses), Xs.detach().cpu().numpy(), ys.detach().cpu().numpy()

    def info_nce_loss(self, features):
        z_a, z_b = features[:self.batch_size], features[:self.batch_size] 
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        I = torch.eye(D, device=self.device)
        c_diff = (c - I).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff = c_diff * 5e-3 * (torch.ones((D, D), device=self.device) - I)
        loss = c_diff.sum()
        return loss
#         labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(self.device)

#         features = F.normalize(features, dim=1)

#         similarity_matrix = torch.matmul(features, features.T)
#         # assert similarity_matrix.shape == (
#         #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
#         # assert similarity_matrix.shape == labels.shape

#         # discard the main diagonal from both: labels and similarities matrix
#         mask = torch.eye(labels.shape[0]).bool().to(self.device)
        
#         labels = labels[~mask].view(labels.shape[0], -1)
        
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#         # assert similarity_matrix.shape == labels.shape
#         # select and combine multiple positives
#         positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

#         # select only the negatives the negatives
#         negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

#         logits = torch.cat([positives, negatives], dim=1)
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

#         logits = logits / self.temperature
#         return logits, labels
    
    def _logits_and_loss(self, X):
        self.mutator.reset()

        features = self.model(X)
#         logits, labels 
        loss = self.info_nce_loss(features)
#         loss = self.loss(logits, labels)
        self._write_graph_status()
        return loss

    def _backward(self, val_X):
        """
        Simple backward with gradient descent
        """
        loss = self._logits_and_loss(val_X)
        params = torch.Tensor([]).to(self.device)
        for param in self.mutator.choices.values():
            params = torch.cat((params, torch.sigmoid(param)))
            loss_0_1 = -F.mse_loss(params, torch.tensor(0.5, requires_grad=False).to(self.device))
        loss += loss_0_1
        loss.backward()
        return loss