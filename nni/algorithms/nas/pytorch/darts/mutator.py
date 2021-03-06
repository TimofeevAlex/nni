# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../../../nni')
from nas.pytorch.mutator import Mutator
from nas.pytorch.mutables import LayerChoice, InputChoice

_logger = logging.getLogger(__name__)


class DartsMutator(Mutator):
    """
    Connects the model in a DARTS (differentiable) way.

    An extra connection is automatically inserted for each LayerChoice, when this connection is selected, there is no
    op on this LayerChoice (namely a ``ZeroOp``), in which case, every element in the exported choice list is ``false``
    (not chosen).

    All input choice will be fully connected in the search phase. On exporting, the input choice will choose inputs based
    on keys in ``choose_from``. If the keys were to be keys of LayerChoices, the top logit of the corresponding LayerChoice
    will join the competition of input choice to compete against other logits. Otherwise, the logit will be assumed 0.

    It's possible to cut branches by setting parameter ``choices`` in a particular position to ``-inf``. After softmax, the
    value would be 0. Framework will ignore 0 values and not connect. Note that the gradient on the ``-inf`` location will
    be 0. Since manipulations with ``-inf`` will be ``nan``, you need to handle the gradient update phase carefully.

    Attributes
    ----------
    choices: ParameterDict
        dict that maps keys of LayerChoices to weighted-connection float tensors.
    """
    def __init__(self, model):
        super().__init__(model)
        self.choices = nn.ParameterDict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length + 1))

    def device(self):
        for v in self.choices.values():
            return v.device

    def sample_search(self):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                result[mutable.key] = torch.sigmoid(self.choices[mutable.key])[:-1]
            elif isinstance(mutable, InputChoice):
                result[mutable.key] = torch.ones(mutable.n_candidates, dtype=torch.bool, device=self.device())
        return result

    def sample_final(self, epoch):
        result = dict()
        edges_max = dict()
        ops_dist = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                op_distr = torch.sigmoid(self.choices[mutable.key])
                max_val, index = torch.max(op_distr[:-1], 0)
                edges_max[mutable.key] = max_val
                ops_dist[mutable.key] = op_distr.detach().cpu()
                result[mutable.key] = F.one_hot(index, num_classes=len(mutable)).view(-1).bool()
        for mutable in self.mutables:
            if isinstance(mutable, InputChoice):
                if mutable.n_chosen is not None:
                    weights = []
                    for src_key in mutable.choose_from:
                        if src_key not in edges_max:
                            _logger.warning("InputChoice.NO_KEY in '%s' is weighted 0 when selecting inputs.", mutable.key)
                        weights.append(edges_max.get(src_key, 0.))
                    weights = torch.tensor(weights)  # pylint: disable=not-callable
                    _, topk_edge_indices = torch.topk(weights, mutable.n_chosen)
                    selected_multihot = []
                    for i, src_key in enumerate(mutable.choose_from):
                        if i not in topk_edge_indices and src_key in result:
                            # If an edge is never selected, there is no need to calculate any op on this edge.
                            # This is to eliminate redundant calculation.
                            result[src_key] = torch.zeros_like(result[src_key])
                        selected_multihot.append(i in topk_edge_indices)
                    result[mutable.key] = torch.tensor(selected_multihot, dtype=torch.bool, device=self.device())  # pylint: disable=not-callable
                else:
                    result[mutable.key] = torch.ones(mutable.n_candidates, dtype=torch.bool, device=self.device())  # pylint: disable=not-callable
                    
        ops = np.array(["maxpool", 
                        "avgpool", 
                        "skipconnect",
                        "sepconv3x3",
                        "sepconv5x5",
                        "dilconv3x3",
                        "dilconv5x5",
                        "zero",])
        ops_dist = pd.DataFrame(ops_dist, index=ops)
        fig, [ax1, ax2] = plt.subplots(2, figsize=(10,10))
        g = sns.countplot(ops[np.argmax(ops_dist.to_numpy(), axis=0)], ax=ax1)
        g.set_xticklabels(g.get_xticklabels(), rotation=45)

        g = sns.heatmap(ops_dist, cmap='coolwarm', ax=ax2)
        g.set_xticklabels(g.get_xticklabels(), rotation=45)
        g.set_yticklabels(g.get_yticklabels(), rotation=45)
        plt.tight_layout()
        plt.savefig('ops_dist/'+str(epoch)+'_ops_distr.png')
        return result
