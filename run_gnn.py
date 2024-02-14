
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data as PYGData
from torch.utils import data
import torch
from os.path import join
import numpy as np
import sklearn.neighbors as skn
import pickle
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('style.mcstyle')


mpl.style.use('style.mcstyle')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# Parse arguments
parser = argparse.ArgumentParser(
    description="Run SBI inference for toy data.")
parser.add_argument('--data', type=str, default='dC100')
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

# CONFIGURATION
dname = f'AMICO{args.data}'
rmax = 0.5
bs = 64
validation_fraction = 0.1


# load data
datapath = join('data/processed', dname)
print('Loading from:', datapath)
x = np.load(join(datapath, 'x_batch.npy'), allow_pickle=True)
theta = np.load(join(datapath, 'theta_batch.npy'), allow_pickle=True)
folds = np.load(join(datapath, 'folds_batch.npy'), allow_pickle=True)

x_train = x[folds != args.fold]
theta_train = theta[folds != args.fold]
x_test = x[folds == args.fold]
theta_test = theta[folds == args.fold]


x = np.concatenate(x_train, axis=0)
theta = np.concatenate(theta_train, axis=0)


# get adjacency matrices


def get_adjacency(x, rmax=1.):
    graph = skn.radius_neighbors_graph(
        x[:, :2], rmax, mode='distance', include_self=False).toarray()
    # graph = skn.kneighbors_graph(
    #     x[:, :2], 8, mode='distance', include_self=False).toarray()
    adj = np.nonzero(graph)
    left = x[adj[0], :3]
    right = x[adj[1], :3]
    dist = np.abs(left - right)
    adj, dist = map(np.array, (adj, dist))
    adj, dist = map(torch.as_tensor, (adj, dist))
    return adj, dist


adj_train, edgattr_train = zip(
    *[get_adjacency(x_, rmax=rmax) for x_ in x_train])
adj_test, edgattr_test = zip(
    *[get_adjacency(x_, rmax=rmax) for x_ in x_test])


# normalize
x_mu, x_std = x.mean(axis=0), x.std(axis=0)

x_train = [torch.Tensor(x_ - x_mu)/x_std for x_ in x_train]
theta_train = torch.Tensor(theta_train)[:, None]
x_test = [torch.Tensor(x_ - x_mu)/x_std for x_ in x_test]
theta_test = torch.Tensor(theta_test)[:, None]
edgattr_train = [e/x_std[:3] for e in edgattr_train]
edgattr_test = [e/x_std[:3] for e in edgattr_test]

# Create pyg datasets
data_train = [
    PYGData(x=x, y=y, edge_index=adj, edge_attr=attr)
    for x, y, adj, attr in zip(x_train, theta_train, adj_train, edgattr_train)
]
data_test = [
    PYGData(x=x, y=y, edge_index=adj, edge_attr=attr)
    for x, y, adj, attr in zip(x_test, theta_test, adj_test, edgattr_test)
]


class GraphData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


data_train = GraphData(data_train)
data_test = GraphData(data_test)


# use pyg's collater
collater = Collater(dataset=data_train)


def collate_fn(batch):
    batch = collater(batch)
    return batch, batch.y


# split train and validation
np.random.seed(1952)
n_train = int((1-validation_fraction)*len(data_train))
permuted_idx = np.random.permutation(len(data_train))
idx_train = permuted_idx[:n_train]
idx_val = permuted_idx[n_train:]

# define train and validation dataloaders
train_loader = data.DataLoader(
    data_train, batch_size=bs, collate_fn=collate_fn,
    sampler=data.SubsetRandomSampler(idx_train)
)
val_loader = data.DataLoader(
    data_train, batch_size=bs, collate_fn=collate_fn,
    sampler=data.SubsetRandomSampler(idx_val)
)
test_loader = data.DataLoader(
    data_test, batch_size=bs, shuffle=False, collate_fn=collate_fn
)


# define an ili loader
train_stage_loader = TorchLoader(train_loader, val_loader)
test_stage_loader = TorchLoader(test_loader, val_loader)

# define an ili trainer
out_dir = f"./saved_models/gnn_npe_{args.data}_f{args.fold}"
runner = InferenceRunner.from_config(
    'configs/inf/gnn.yaml', device=device,
    out_dir=out_dir
)
posterior_ensemble, summary = runner(loader=train_stage_loader)


# plot loss function
f = plt.figure()
for i, x in enumerate(summary):
    plt.plot(x['training_log_probs'], label=f'train {i}', c=f'C{i}')
    plt.plot(x['validation_log_probs'],
             label=f'val {i}', c=f'C{i}', ls='--')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Log probability')
f.savefig(f"{out_dir}/loss.png")

# run metrics
pfile = join(runner.out_dir, 'posterior.pkl')
with open(pfile, 'rb') as f:
    posterior_ensemble = pickle.load(f)


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
    x=data_test, theta=theta_test[..., 0]
)
