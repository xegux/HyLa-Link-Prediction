#!/usr/bin/env python3
# import sys
# sys.path.append("..")
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import geoopt
import torch
import logging
import argparse
import json
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
from hype import MANIFOLDS, MODELS, build_model, get_model
from hype.hyla_utils import sgc_precompute, acc_f1, load_data, load_reddit_data, load_data_lp
import torch.nn.functional as F
import timeit
import gc
from sklearn.metrics import roc_auc_score, average_precision_score
from hype.models import LP

def generate_ckpt(opt, model, path):
    checkpoint = LocalCheckpoint(
        path,
        include_in_all={'conf': vars(opt)},
        start_fresh=opt.fresh
    )
    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']
    return checkpoint

def encode(x, adj):
        if encode_graph:
            i = (x, adj)
            output, _ = layers.forward(i)
        else:
            output = layers.forward(x)
        return output

def decode(h, idx):
    if manifold_name == 'Euclidean':
        h = manifold.normalize(h)
    emb_in = h[idx[:, 0], :]
    emb_out = h[idx[:, 1], :]
    sqdist = manifold.sqdist(emb_in, emb_out, c)
    probs = dc.forward(sqdist)
    return probs

def compute_metrics(embeddings, data, split):
    if split == 'train':
        edges_false = data[f'{split}_edges_false'][np.random.randint(0, nb_false_edges, nb_edges)]
    else:
        edges_false = data[f'{split}_edges_false']
    pos_scores = decode(embeddings, data[f'{split}_edges'])
    neg_scores = decode(embeddings, edges_false)
    loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
    loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
    if pos_scores.is_cuda:
        pos_scores = pos_scores.cpu()
        neg_scores = neg_scores.cpu()
    labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
    preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
    roc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    metrics = {'loss': loss, 'roc': roc, 'ap': ap}
    return metrics

def init_metric_dict():
    return {'roc': -1, 'ap': -1}

def has_improved(m1, m2):
    return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

def train(model_f,
          model_c,
          optimizer_f,
          optimizer_c,
          optim_b,
          data,
          opt,
          log,
          progress=False,
          ckps=None):
    model_f.train()
    model_c.train()
    val_roc_best = 0.0
    train_roc_best = 0.0
    for epoch in range(opt.epoch_start, opt.epochs):
        t_start = timeit.default_timer()
        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optim_b.zero_grad()
        HyLa_features = model_f()
        HyLa_features = torch.mm(
            data['features_train'].to(opt.device), HyLa_features)
        predictions = model_c(HyLa_features)
        del HyLa_features
        # loss = F.cross_entropy(
        #     predictions, data['labels'][data['idx_train']].to(opt.device))

        embeddings = encode(data['features'], data['adj_train_norm'])
        train_metrics = compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        optimizer_f.step()
        optimizer_c.step()
        optim_b.step()

        train_roc = train_metrics['roc']
        val_roc = compute_metrics(embeddings, data, 'val')['roc']
        
        if val_roc > val_roc_best:
            val_roc_best = val_roc
            if ckps is not None:
                ckps[0].save({
                    'model': model_f.state_dict(),
                    'epoch': epoch,
                    'val_roc_best': val_roc_best,
                })
                ckps[1].save({
                    'model': model_c.state_dict(),
                    'epoch': epoch,
                    'val_roc_best': val_roc_best,
                })

        if train_roc > train_roc_best:
            train_roc_best = train_roc

        loss = train_metrics['loss'].cpu().item()

        if progress:
            log.info(
                'running stats: {'
                f'"epoch": {epoch}, '
                f'"elapsed": {timeit.default_timer()-t_start:.2f}, '
                f'"train_roc": {train_roc*100.0:.2f}%, '
                f'"val_roc": {val_roc*100.0:.2f}%, '
                f'"loss_c": {loss.cpu().item():.4f}, '
                '}'
            )
        gc.collect()
        torch.cuda.empty_cache()
    return train_roc, train_roc_best, val_roc, val_roc_best

def main():
    parser = argparse.ArgumentParser(
        description='Train HyLa-LP for link prediction tasks')
    parser.add_argument('-checkpoint', action='store_true', default=False)
    parser.add_argument('-task', type=str, default='nc', help='learning task')
    parser.add_argument('-dataset', type=str, required=True,
                        help='Dataset identifier [cora|disease_nc|pubmed|citeseer|reddit|airport]')
    parser.add_argument('-he_dim', type=int, default=2,
                        help='Hyperbolic Embedding dimension')
    parser.add_argument('-hyla_dim', type=int, default=100,
                        help='HyLa feature dimension')
    parser.add_argument('-order', type=int, default=2,
                        help='order of adjaceny matrix in LP precomputation')
    parser.add_argument('-manifold', type=str, default='poincare',
                        choices=MANIFOLDS.keys(), help='model of hyperbolic space')
    parser.add_argument('-model', type=str, default='hyla',
                        choices=MODELS.keys(), help='feature model class, hyla|rff')
    parser.add_argument('-lr_e', type=float, default=0.1,
                        help='Learning rate for hyperbolic embedding')
    parser.add_argument('-lr_c', type=float, default=0.1,
                        help='Learning rate for LP')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('-strategy', type=int, default=0,
                        help='Epochs of burn in, some advanced definition')
    parser.add_argument('-eval_each', type=int, default=1,
                        help='Run evaluation every n-th epoch')
    parser.add_argument('-fresh', action='store_true', default=False,
                        help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int,
                        help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-seed', default=43, type=int, help='random seed')
    parser.add_argument('-sparse', default=True, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument(
        '-lre_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-optim_type', choices=['adam', 'sgd'], default='adam',
                        help='optimizer used for the LP model')
    parser.add_argument(
        '-metric', choices=['acc', 'f1'], default='acc', help='what metrics to report')
    parser.add_argument('-lambda_scale', type=float, default=0.07,
                        help='scale of lambdas when generating HyLa features')
    parser.add_argument('-inductive', action='store_true',
                        default=False, help='inductive training, used for reddit.')
    parser.add_argument('-use_feats', action='store_true', default=False,
                        help='whether embed in the feature level, otherwise node level')
    parser.add_argument('-tuned', action='store_true',
                        default=False, help='whether use tuned hyper-parameters')
    opt = parser.parse_args()

    if opt.tuned:
        with open(f'{currentdir}/hyper_parameters_{opt.he_dim}d.json',) as f:
            hyper_parameters = json.load(f)[opt.dataset]
        # for lp: input is 2 nodes instead of 1 node
        opt.he_dim = hyper_parameters['he_dim']
        opt.hyla_dim = hyper_parameters['hyla_dim']
        opt.order = hyper_parameters['order']
        opt.lambda_scale = hyper_parameters['lambda_scale']
        opt.lr_e = hyper_parameters['lr_e']
        opt.lr_c = hyper_parameters['lr_c']
        opt.epochs = hyper_parameters['epochs']

    opt.metric = 'f1' if opt.dataset == 'reddit' else 'acc'
    opt.epoch_start = 0
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    opt.split_seed = opt.seed
    opt.progress = not opt.quiet

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('HyLa')
    logging.basicConfig(
        level=log_level, format='%(message)s', stream=sys.stdout)

    # set default tensor type
    torch.set_default_tensor_type('torch.DoubleTensor')
    # set device
    opt.device = torch.device(
        f'cuda:{opt.gpu}' if opt.gpu >= 0 and torch.cuda.is_available() else 'cpu')

    # here for loading adj things for classification task
    data_path = f'{currentdir}/datasets/' + opt.dataset + '/'
    if opt.dataset in ['cora', 'disease_nc', 'pubmed', 'citeseer', 'airport']:
        data = load_data(opt, data_path)
    # elif opt.dataset in ['reddit']:
    #     data = load_reddit_data_lp(data_path)
    else:
        raise NotImplemented

    # setup dataset parameters and setting
    if opt.use_feats or opt.inductive:
        if opt.progress:
            log.info(f'hyperbolic Laplacian features used in the feature level ...')
        feature_dim = data['features'].size(1)
    else:
        if opt.progress:
            log.info(f'hyperbolic Laplacian features used in the node level ...')
        feature_dim = data['adj_train'].size(1)
    
    if opt.progress:
        log.info(
            f'info about the data, training set size :{len(data["adj_train"])}')
        log.info(
            f'size of original feature matrix: {data["features"].size()}')
        log.info('precomputing features')

    if opt.inductive:
        features = data['features']
        data['features'], _ = sgc_precompute(
            data['adj_all'], features, opt.order)
        # data['features_train'], nonzero_perc = sgc_precompute(
        #     data['adj_train'], features[data['idx_train']], opt.order)
    else:
        if not opt.use_feats:
            features = data['adj_train'].to_dense()
            data['features'], nonzero_perc = sgc_precompute(
                data['adj_train'], features, opt.order-1)
        else:
            features = data['features'].to_dense()
            data['features'], nonzero_perc = sgc_precompute(
                data['adj_train'], features, opt.order)
        data['features_train'] = data['features']
        #data['features_train'] = data['features'][data['idx_train']]
    if opt.progress:
        log.info(
            f'nonzero_perc during adjacency matrix precomputations: {nonzero_perc}%')

    # build feature models and setup optimizers
    model_f = build_model(opt, feature_dim).to(opt.device)
    if opt.lre_type == 'scale':
        opt.lr_e = opt.lr_e * len(data['idx_train'])
    if opt.manifold == 'euclidean':
        #         optimizer_f = torch.optim.Adam(model_f.parameters(), lr=opt.lr_e)# weight_decay=1.3e-5
        optimizer_f = torch.optim.SGD(model_f.parameters(), lr=opt.lr_e)
    elif opt.manifold == 'poincare':
        optimizer_f = RiemannianSGD(model_f.optim_params(), lr=opt.lr_e)
        optim_b = geoopt.optim.RiemannianAdam([model_f.boundary], lr=1e-3)

    # build link prediction models and setup optimizers
    model_c = LP(len(data['train_edges_false']), 
                        len(data['train_edges'])).to(opt.device)
    if opt.optim_type == 'sgd':
        optimizer_c = torch.optim.SGD(model_c.parameters(), lr=opt.lr_c)
    elif opt.optim_type == 'adam':
        # , weight_decay=1.0e-4)
        optimizer_c = torch.optim.Adam(model_c.parameters(), lr=opt.lr_c)
    else:
        raise NotImplementedError

    ckps = None
    if opt.checkpoint:
        # setup checkpoint
        ckp_fm = generate_ckpt(
            opt, model_f, f'{currentdir}/datasets/' + opt.dataset + '/fm.pt')
        ckp_cm = generate_ckpt(
            opt, model_c, f'{currentdir}/datasets/' + opt.dataset + '/cm.pt')
        ckps = (ckp_fm, ckp_cm)
    t_start_all = timeit.default_timer()
    train_roc, train_roc_best, val_roc, val_roc_best = train(
        model_f, model_c, optimizer_f, optimizer_c, optim_b,
        data, opt, log, progress=opt.progress, ckps=ckps)
    if opt.progress:
        log.info(f'TOTAL ELAPSED: {timeit.default_timer()-t_start_all:.2f}')
    if opt.checkpoint and ckps is not None:
        state_fm = ckps[0].load()
        state_cm = ckps[1].load()
        model_f.load_state_dict(state_fm['model'])
        model_c.load_state_dict(state_cm['model'])
        if opt.progress:
            log.info(
                f'early stopping, loading from epoch: {state_fm["epoch"]} with val_acc_best: {state_fm["val_acc_best"]}')
            
    embeddings = encode(data['features'], data['adj_train_norm'])
    test_roc = compute_metrics(embeddings, data, 'test')['roc']
#     test_acc_threshold = {'cora': 0, 'disease_nc': 0, 'pubmed': 0, 'citeseer': 0, 'reddit': 0, 'airport': 0}
#     test_acc_threshold = {'cora': 82, 'disease_nc': 80, 'pubmed': 80, 'citeseer': 71, 'reddit': 93.5, 'airport': 80}
#     if test_acc * 100.0 > test_acc_threshold[opt.dataset]:
    log.info(
        f'"|| last train_roc": {train_roc*100.0:.2f}%, '
        f'"|| best train_roc": {train_roc_best*100.0:.2f}%, '
        f'"|| last val_roc": {val_roc*100.0:.2f}%, '
        f'"|| best val_roc": {val_roc_best*100.0:.2f}%, '
        f'"|| test_roc": {test_roc*100.0:.2f}%.'
    )


if __name__ == '__main__':
    main()
