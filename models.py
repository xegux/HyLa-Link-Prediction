import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import math
import numpy as np
from .hyla_utils import PoissonKernel, sample_boundary, measure_tensor_size
import dgl


class HyLa(nn.Module):
    def __init__(self, manifold, dim, size, HyLa_fdim, scale=0.1, sparse=False, **kwargs):
        super(HyLa, self).__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.manifold.init_weights(self.lt)
        self.dim = dim
        self.Lambdas = scale * torch.randn(HyLa_fdim)
        self.boundary = nn.Parameter(sample_boundary(
            HyLa_fdim, self.dim, cls='RandomUniform'))
        self.bias = 2 * np.pi * torch.rand(HyLa_fdim)
        # self.fc1 = nn.Linear(HyLa_fdim, 1)
        # self.fc2 = nn.Linear(HyLa_fdim, 1)

    def forward(self):
        with torch.no_grad():
            e_all = self.manifold.normalize(self.lt.weight)

        PsK = PoissonKernel(e_all, self.boundary.to(e_all.device))

        angles = self.Lambdas.to(e_all.device)/2.0 * torch.log(PsK)
        eigs = torch.cos(angles + self.bias.to(e_all.device)
                         ) * torch.sqrt(PsK)**(self.dim-1)
        # eigs = eigs.view(-1, 1)
        # eigs1 = self.fc1(eigs)
        # eigs2 = self.fc2(eigs)
        # return eigs1, eigs2
        return eigs

    def optim_params(self):
        return [{
            'params': self.lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]


class RFF(nn.Module):
    def __init__(self, manifold, dim, size, HyLa_fdim, scale=0.1, sparse=False, **kwargs):
        super(RFF, self).__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.manifold.init_weights(self.lt)
        self.norm = 1. / np.sqrt(dim)
        self.Lambdas = nn.Parameter(torch.from_numpy(np.random.normal(
            loc=0, scale=scale, size=(dim, HyLa_fdim))), requires_grad=False)
        self.bias = nn.Parameter(torch.from_numpy(np.random.uniform(
            0, 2 * np.pi, size=HyLa_fdim)), requires_grad=False)

    def forward(self):
        with torch.no_grad():
            e_all = self.manifold.normalize(self.lt.weight)
        features = self.norm * \
            np.sqrt(2) * torch.cos(e_all @ self.Lambdas + self.bias)
        return features

    def optim_params(self):
        return [{
            'params': self.lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]

from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
import dgl.function as fn

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        
# given 2 nodes, find probability of a link between them by computing corresponding hyla features (eigs1, eigs2)
# use binary cross entropy loss to maximize link likelihood <eigs1, eigs2> --> 1 and minimize link likelihood <eigs1, eigs2>--> 0