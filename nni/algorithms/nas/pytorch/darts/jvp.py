from __future__ import print_function
import scipy.sparse.linalg as linalg
import torch
from torch import autograd
import numpy as np


def tensorify_grad(grad, ignore_none=False):
    if isinstance(grad, (list, tuple)):
        grad = list(grad)
        if ignore_none:
            grad = [g for g in grad if g is not None]
        for i, g in enumerate(grad):
            if g is None:
                raise Exception("One gradient was None. Remember to call backwards.")
            grad[i] = g.view(-1)
        return torch.cat(grad)
    elif isinstance(grad, torch.Tensor):
        return grad.view(-1)
    

class JacobianVectorProduct(linalg.LinearOperator):
    def __init__(self, grad, params, force_numpy=False):
        self.grad = tensorify_grad(grad)
        self.shape = (self.grad.size(0), self.grad.size(0))
        self.dtype = np.dtype('float32')
        self.params = params
        self.force_numpy = force_numpy

    def _matvec(self, v, force_cpu_output=True):
        # TODO: avoid copying?
        v = torch.tensor(v)
        if self.grad.is_cuda and not v.is_cuda:
            v = v.cuda()
        grad_vector_product = torch.dot(self.grad, v)
        hv = autograd.grad(grad_vector_product, self.params, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        hv = torch.cat(_hv)
        if force_cpu_output:
            hv = hv.cpu()
        if self.force_numpy:
            return hv.numpy()
        else:
            return hv