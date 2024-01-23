
from os.path import join
import numpy as np
import sklearn.neighbors as skn
import pickle
import argparse

import torch
from torch.utils import data
from torch_geometric.data import Data as PYGData
from torch_geometric.loader.dataloader import Collater

from ili.inference import InferenceRunner
from ili.validation import ValidationRunner
from ili.dataloaders import TorchLoader
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# Parse arguments
parser = argparse.ArgumentParser(
    description="Run SBI inference for toy data.")
parser.add_argument('--data', type=str, default='dC100')
args = parser.parse_args()

# CONFIGURATION
dname = f'AMICO{args.data}'
rmax = 0.5
bs = 16
validation_fraction = 0.1


# load data
datapath = join('data/processed', dname)
print('Loading from:', datapath)
x_train = np.load(join(datapath, 'x_batch_train.npy'), allow_pickle=True)
theta_train = np.load(
    join(datapath, 'theta_batch_train.npy'), allow_pickle=True)
x_test = np.load(join(datapath, 'x_batch_test.npy'), allow_pickle=True)
theta_test = np.load(join(datapath, 'theta_batch_test.npy'), allow_pickle=True)

x = np.concatenate(x_train, axis=0)
theta = np.concatenate(theta_train, axis=0)


# get adjacency matrices


def get_adjacency(x, rmax=1.2):
    graph = skn.radius_neighbors_graph(
        x[:, :2], rmax, mode='distance', include_self=True).toarray()
    adj = np.nonzero(graph)
    return torch.Tensor((adj))


adj_train = [get_adjacency(x_, rmax=rmax) for x_ in x_train]
adj_test = [get_adjacency(x_, rmax=rmax) for x_ in x_test]


# normalize
x_mu, x_std = x.mean(axis=0), x.std(axis=0)

x_train = [torch.Tensor(x_ - x_mu)/x_std for x_ in x_train]
theta_train = torch.Tensor(theta_train)[:, None]
x_test = [torch.Tensor(x_ - x_mu)/x_std for x_ in x_test]
theta_test = torch.Tensor(theta_test)[:, None]

# Create pyg datasets
data_train = [
    PYGData(x=x, y=y, edge_index=adj)
    for x, y, adj in zip(x_train, theta_train, adj_train)
]
data_test = [
    PYGData(x=x, y=y, edge_index=adj)
    for x, y, adj in zip(x_test, theta_test, adj_test)
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
runner = InferenceRunner.from_config(
    'configs/inf/gnn.yaml', device=device,
    out_dir=f"./saved_models/gnn_npe_{args.data}"
)
posterior_ensemble, summaries = runner(loader=train_stage_loader)


# # define a validation runner
# val_runner = ValidationRunner.from_config(
#     'configs/val/gnn.yaml', device=device
# )
# val_runner(loader=test_stage_loader)

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
