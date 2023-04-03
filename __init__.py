#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import manifolds
import argparse
from . import models

MANIFOLDS = {
    'lorentz': manifolds.LorentzManifold,
    'poincare': manifolds.PoincareManifold,
    'euclidean': manifolds.EuclideanManifold,
}

MODELS = {
    'hyla': models.HyLa,
    'rff': models.RFF,
    'lp': models.GraphSAGE,
}

def build_model(opt, N):
    if isinstance(opt, argparse.Namespace):
        opt = vars(opt)
    manifold = MANIFOLDS[opt['manifold']](K=None)
    return MODELS[opt['model']](
        manifold,
        dim=opt['he_dim'],
        size=N,
        HyLa_fdim=opt['hyla_dim'],
        scale=opt['lambda_scale'],
        sparse=opt['sparse'],
    )

def get_model(model_opt, in_feats, h_feats, adj=None, dropout=0.0):
    # if model_opt == "SGC":
    #     model = models.SGC(nfeats=nfeats,
    #                 hfeats=hfeats)
    if model_opt == "GraphSAGE":
        model = models.GraphSAGE(in_feats=in_feats, h_feats=h_feats)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))
    return model
