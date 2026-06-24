import os
import json
import torch
from os.path import join
import numpy as np
import sklearn.neighbors as skn
import pickle
import argparse
import yaml

import optuna

from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner
from ili.utils import load_from_config, load_nde_lampe
from tools.networks import GATNetwork
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data as PYGData
from torch.utils import data

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('style.mcstyle')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# Parse arguments
parser = argparse.ArgumentParser(
    description="Run SBI inference with optuna hyperparameter tuning.")
parser.add_argument('--data', type=str, default='dC100')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--from_scratch', type=int, default=1)
parser.add_argument('--n_trials', type=int, default=50,
                    help='Number of optuna trials for the architecture search.')
parser.add_argument('--final_repeats', type=int, default=3,
                    help='Number of flows in the final trained ensemble.')
args = parser.parse_args()

# CONFIGURATION
dname = f'APR24{args.data}'
savename = 'oct02_optuna'
bs = 64                      # fixed batch size (not tuned)
validation_fraction = 0.1
in_channels = 3             # node features (ignore richness dependence)

# specify directories
out_dir = f"./saved_models/{savename}_gnn_npe_{args.data}_f{args.fold}"
os.makedirs(out_dir, exist_ok=True)
if args.from_scratch == 0 and os.path.exists(f"{out_dir}/posterior_samples.npy"):
    print('Skipping inference, posterior already exists.')
    exit()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
datapath = join('data/processed', dname)
print('Loading from:', datapath)
x = np.load(join(datapath, 'x_batch.npy'), allow_pickle=True)
theta = np.load(join(datapath, 'theta_batch.npy'), allow_pickle=True)
folds = np.load(join(datapath, 'folds_batch.npy'), allow_pickle=True)

x = np.array([t[:, :in_channels] for t in x], dtype=object)  # ignore richness

x_train = x[folds != args.fold]
theta_train_raw = theta[folds != args.fold]
x_test = x[folds == args.fold]
theta_test_raw = theta[folds == args.fold]

# normalization stats from the training nodes (independent of graph topology)
x_all = np.concatenate(x_train, axis=0)
x_mu, x_std = x_all.mean(axis=0), x_all.std(axis=0)

# pre-normalize targets once (shared across all graph configs)
theta_train_t = torch.Tensor(theta_train_raw)[:, None]
theta_test_t = torch.Tensor(theta_test_raw)[:, None]


# ---------------------------------------------------------------------------
# Graph construction (with caching per (method, param))
# ---------------------------------------------------------------------------
def get_adjacency(x_, method, param):
    """Build an adjacency + edge-attribute tensor for a single cluster."""
    if method == 'radius':
        graph = skn.radius_neighbors_graph(
            x_[:, :2], param, mode='distance', include_self=False).toarray()
    elif method == 'knn':
        # clamp k to a valid value for small clusters
        k = int(min(param, max(1, len(x_) - 1)))
        graph = skn.kneighbors_graph(
            x_[:, :2], k, mode='distance', include_self=False).toarray()
    else:
        raise ValueError(f'Unknown graph method: {method}')
    adj = np.nonzero(graph)
    left = x_[adj[0], :in_channels]
    right = x_[adj[1], :in_channels]
    dist = np.abs(left - right)
    adj, dist = map(np.array, (adj, dist))
    adj, dist = map(torch.as_tensor, (adj, dist))
    return adj, dist


class GraphData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


_dataset_cache = {}


def build_datasets(method, param):
    """Build normalized PyG train/test datasets for a graph configuration.

    Cached per (method, param) so trials reusing a topology pay no rebuild
    cost. edge_attr is always stored; whether the model uses it is controlled
    separately by the GATNetwork's edge_attr flag.
    """
    key = (method, round(float(param), 4))
    if key in _dataset_cache:
        return _dataset_cache[key]

    adj_train, edgattr_train = zip(
        *[get_adjacency(x_, method, param) for x_ in x_train])
    adj_test, edgattr_test = zip(
        *[get_adjacency(x_, method, param) for x_ in x_test])

    x_train_n = [torch.Tensor(x_ - x_mu) / x_std for x_ in x_train]
    x_test_n = [torch.Tensor(x_ - x_mu) / x_std for x_ in x_test]
    edgattr_train_n = [e / x_std[:in_channels] for e in edgattr_train]
    edgattr_test_n = [e / x_std[:in_channels] for e in edgattr_test]

    data_train = GraphData([
        PYGData(x=xx, y=yy, edge_index=adj, edge_attr=attr)
        for xx, yy, adj, attr in zip(
            x_train_n, theta_train_t, adj_train, edgattr_train_n)
    ])
    data_test = GraphData([
        PYGData(x=xx, y=yy, edge_index=adj, edge_attr=attr)
        for xx, yy, adj, attr in zip(
            x_test_n, theta_test_t, adj_test, edgattr_test_n)
    ])

    _dataset_cache[key] = (data_train, data_test)
    return data_train, data_test


# fixed train/validation split over the training clusters (shared across
# graph configs, since the number of training clusters is constant)
np.random.seed(1952)
n_clusters = len(x_train)
n_tr = int((1 - validation_fraction) * n_clusters)
_perm = np.random.permutation(n_clusters)
idx_train, idx_val = _perm[:n_tr], _perm[n_tr:]


def make_loaders(data_train, data_test):
    collater = Collater(dataset=data_train)

    def collate_fn(batch):
        batch = collater(batch)
        return batch, batch.y

    train_loader = data.DataLoader(
        data_train, batch_size=bs, collate_fn=collate_fn,
        sampler=data.SubsetRandomSampler(idx_train))
    val_loader = data.DataLoader(
        data_train, batch_size=bs, collate_fn=collate_fn,
        sampler=data.SubsetRandomSampler(idx_val))
    test_loader = data.DataLoader(
        data_test, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model / runner construction
# ---------------------------------------------------------------------------
# Load the prior from the existing config so it stays in sync with run_gnn.py.
with open('configs/inf/gnn.yaml', 'r') as fd:
    _base_config = yaml.safe_load(fd)
_prior_config = _base_config['prior']
_prior_config['args']['device'] = str(device)
prior = load_from_config(_prior_config)


def build_nets(params, repeats):
    """Build a list of lampe NPE net constructors for the given config."""
    gcn_width = params['gcn_width']
    out_channels = params['out_channels']
    embedding_net = GATNetwork(
        in_channels=in_channels,
        gcn_channels=[gcn_width, gcn_width],
        gcn_heads=[4, 1],
        dense_channels=[gcn_width, out_channels],
        out_channels=out_channels,
        drop_p=0.1,
        edge_attr=params['edge_attr'],
    )
    return load_nde_lampe(
        model='nsf',
        embedding_net=embedding_net,
        device=str(device),
        x_normalize=False,
        theta_normalize=True,
        hidden_features=params['hidden_features'],
        num_transforms=params['num_transforms'],
        repeats=repeats,
    )


def build_runner(params, runner_out_dir, repeats):
    nets = build_nets(params, repeats)
    train_args = dict(
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        training_batch_size=bs,
        max_epochs=params['max_epochs'],
        early_stopping=False,        # max_epochs is the training-length knob
        clip_max_norm=5,
        validation_fraction=validation_fraction,
    )
    return InferenceRunner.load(
        backend='lampe', engine='NPE', prior=prior,
        nets=nets, train_args=train_args,
        out_dir=runner_out_dir, device=str(device),
    )


def suggest_params(trial):
    """Sample one hyperparameter configuration.

    Ordinal numeric knobs use suggest_int/suggest_float so the TPE sampler can
    exploit ordering (concentrate near good values) rather than treating them
    as unordered categories. Power-of-two knobs are encoded via the exponent
    to keep the original {8,16,32}-style grid without enlarging the space.
    Genuinely unordered knobs (edge_attr, graph_method) stay categorical.
    """
    params = {
        # flow
        'hidden_features': 2 ** trial.suggest_int('log2_hidden_features', 3, 5),
        'num_transforms': trial.suggest_int('num_transforms', 2, 6, step=2),
        # GAT embedding net
        'gcn_width': 2 ** trial.suggest_int('log2_gcn_width', 3, 5),
        'out_channels': 2 ** trial.suggest_int('log2_out_channels', 2, 4),
        'edge_attr': trial.suggest_categorical('edge_attr', [True, False]),
        # training
        'learning_rate': trial.suggest_float(
            'learning_rate', 1e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_categorical(
            'weight_decay', [0.0, 1e-5, 1e-4]),
        'max_epochs': trial.suggest_int('max_epochs', 50, 200, step=50),
    }
    # graph topology (conditional parameter); kept on a discrete grid via step
    # so the adjacency cache stays bounded to a few rebuilds.
    params['graph_method'] = trial.suggest_categorical(
        'graph_method', ['radius', 'knn'])
    if params['graph_method'] == 'radius':
        params['graph_param'] = trial.suggest_float(
            'rmax', 0.3, 0.8, step=0.1)
    else:
        params['graph_param'] = trial.suggest_int('k', 4, 16, step=2)
    return params


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
# Note: LampeRunner does not expose a per-epoch callback, so we cannot report
# intermediate values for pruning. We rely on the TPE sampler for efficiency
# and run each trial to its sampled max_epochs.
def objective(trial):
    params = suggest_params(trial)

    data_train, _ = build_datasets(params['graph_method'],
                                   params['graph_param'])
    train_loader, val_loader, _ = make_loaders(data_train, None)
    loader = TorchLoader(train_loader, val_loader)

    # single flow during search (out_dir=None => nothing written to disk)
    runner = build_runner(params, runner_out_dir=None, repeats=1)
    try:
        _, summary = runner(loader=loader, seed=1952, verbose=False)
    except Exception as e:
        print(f'Trial {trial.number} failed: {e}')
        return float('-inf')

    val_logprobs = summary[0]['validation_log_probs']
    if len(val_logprobs) == 0 or not np.isfinite(val_logprobs[-1]):
        return float('-inf')
    # mean over the last few epochs; aligns with the last-epoch weights that
    # the runner keeps when early_stopping is off, and reduces epoch noise.
    return float(np.mean(val_logprobs[-3:]))


# ---------------------------------------------------------------------------
# Run the study
# ---------------------------------------------------------------------------
storage = f"sqlite:///{out_dir}/optuna_study.db"
study_name = f"{savename}_{args.data}_f{args.fold}"
study = optuna.create_study(
    study_name=study_name, storage=storage,
    direction='maximize', load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=1952),
)

n_remaining = args.n_trials - len([
    t for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE])
if n_remaining > 0:
    study.optimize(objective, n_trials=n_remaining)

print('Best validation log-prob:', study.best_value)
print('Best params:', study.best_params)

# persist the best configuration
best_params = dict(study.best_params)
with open(f"{out_dir}/best_params.json", 'w') as fd:
    json.dump({'best_value': study.best_value, 'params': best_params}, fd,
              indent=2)


# ---------------------------------------------------------------------------
# Train the final ensemble with the best configuration
# ---------------------------------------------------------------------------
# resolve the best config through suggest_params (decodes the 2** exponents
# and the conditional graph parameter consistently with the search)
final_params = suggest_params(optuna.trial.FixedTrial(best_params))

data_train, data_test = build_datasets(
    final_params['graph_method'], final_params['graph_param'])
train_loader, val_loader, test_loader = make_loaders(data_train, data_test)
train_stage_loader = TorchLoader(train_loader, val_loader)

runner = build_runner(final_params, runner_out_dir=out_dir,
                      repeats=args.final_repeats)
posterior_ensemble, summary = runner(loader=train_stage_loader, seed=1952)


# plot loss function
f = plt.figure()
for i, s in enumerate(summary):
    plt.plot(s['training_log_probs'], label=f'train {i}', c=f'C{i}')
    plt.plot(s['validation_log_probs'],
             label=f'val {i}', c=f'C{i}', ls='--')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Log probability')
f.savefig(f"{out_dir}/loss.png")

# run metrics
pfile = join(runner.out_dir, 'posterior.pkl')
with open(pfile, 'rb') as fd:
    posterior_ensemble = pickle.load(fd)


# SinglePosterior
ind = np.random.choice(len(data_test))
x_ = data_test[ind]
y_ = x_.y
metric = PlotSinglePosterior(
    num_samples=1000, sample_method='direct',
    labels=['logM200'], out_dir=runner.out_dir
)
fig = metric(
    posterior=posterior_ensemble,
    x_obs=x_, theta_fid=y_
)

# PosteriorCoverage
metric = PosteriorCoverage(
    num_samples=1000, sample_method='direct',
    labels=['logM200'], out_dir=runner.out_dir,
    plot_list=["coverage", "histogram", "predictions", "tarp"],
    save_samples=True
)
fig = metric(
    posterior=posterior_ensemble,
    x=data_test, theta=theta_test_t[..., 0]
)
