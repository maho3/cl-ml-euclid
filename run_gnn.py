import os
from os.path import join
import numpy as np
import sklearn.neighbors as skn
import tqdm
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tools.embedding import DenseNetwork, GATNetwork, SAGENetwork
from ili.utils import Uniform, IndependentTruncatedNormal

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# CONFIGURATION
dname = 'FS2dC100'  # 'Lorenzo' #
rmax = 1.5

patience = 30
lr = 1e-5
prior = Uniform(low=0.1, high=1.0)
proposal = IndependentTruncatedNormal(
    low=0.1, high=1.0, loc=0.5, scale=0.1)

outdir = './saved_models/gnn_npe'
os.makedirs(outdir, exist_ok=True)

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
    return tuple(adj)


adj_train = [get_adjacency(x_, rmax=rmax) for x_ in x_train]
adj_test = [get_adjacency(x_, rmax=rmax) for x_ in x_test]


# normalize
# normalize
x_mu, x_std = x.mean(axis=0), x.std(axis=0)
t_mu, t_std = theta.mean(axis=0), theta.std(axis=0)

x_train = [(x_ - x_mu)/x_std for x_ in x_train]
theta_train = [(t_ - t_mu)/t_std for t_ in theta_train]
x_test = [(x_ - x_mu)/x_std for x_ in x_test]
theta_test = [(t_ - t_mu)/t_std for t_ in theta_test]

# convert to arrays
x_train = np.array(x_train, dtype=object)
theta_train = np.array(theta_train)
adj_train = np.array(adj_train, dtype=object)
x_test = np.array(x_test, dtype=object)
theta_test = np.array(theta_test)
adj_test = np.array(adj_test, dtype=object)

# define dataset class


class GraphData(Dataset):
    def __init__(self, x_batch, theta_batch, adj_batch):
        self.x_batch = x_batch
        self.theta_batch = theta_batch
        self.adj_batch = adj_batch

    def __len__(self):
        return len(self.x_batch)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_batch[idx], dtype=torch.float)
        theta = torch.tensor(self.theta_batch[idx], dtype=torch.float)
        edge_index = torch.tensor(
            np.stack(self.adj_batch[idx]))  # .clone().detach()

        theta = theta[0:1]  # only mass
        return x, edge_index, theta


# define train and test dataloaders
train_dataset = GraphData(x_train, theta_train, adj_train)
test_dataset = GraphData(x_test, theta_test, adj_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# define model
model = GATNetwork(
    in_channels=3, gcn_channels=[8, 16, 16, 16, 32], gcn_heads=[4, 4, 4, 4, 1],
    dense_channels=[32, 32, 16], out_channels=2)
model.to(device)

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr, weight_decay=1e-1)


def criterion(pred, true, reduction='mean'):
    pred, true = torch.atleast_2d(pred), torch.atleast_2d(true)
    # return F.mse_loss(pred, true, reduction=reduction)
    return F.mse_loss(pred[:, 1], (true[:, 0]-pred[:, 0])**2, reduction=reduction)


def train(model, optimizer, loader, device, bs=32, verbose=False):
    model.train()

    loss_all = 0
    train_iter = iter(loader)
    for i in tqdm.tqdm(range(len(loader)//bs+1), disable=not verbose):
        if i*bs >= len(loader):
            break
        optimizer.zero_grad()
        loss = torch.tensor(0, dtype=torch.float, device=device)
        for j in range(bs):
            try:
                xi, edgei, thetai = next(train_iter)
            except StopIteration:
                break
            xi, edgei, thetai = xi[0], edgei[0], thetai[0]
            out = model(xi, edgei)
            loss += criterion(out, thetai, reduction='sum')
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def test(model, loader):
    model.eval()
    pred = torch.zeros((len(loader), 2), dtype=torch.float, device='cpu')
    true = torch.zeros((len(loader), 1), dtype=torch.float, device='cpu')
    with torch.no_grad():
        for i, (xi, edgei, thetai) in enumerate(loader):
            xi, edgei, thetai = xi[0], edgei[0], thetai[0]
            out = model(xi, edgei)
            pred[i] = out
            true[i] = thetai
    return true, pred


def predict(model, loader):
    true, pred = test(model, loader)
    true = true*t_std + t_mu
    pred[:, 0] = pred*t_std + t_mu
    pred[:, 1] = pred[:, 1]*(t_std**2)
    return true, pred


# train the model
min_change = 1e-3

trrec = []
terec = []
wait = 0
valoss_min = np.inf
for epoch in range(1000):
    train_loss = train(model, optimizer, train_loader, device, verbose=True)

    # test the model
    true, pred = test(model, test_loader)
    test_loss = criterion(pred, true, reduction='mean').item()

    trrec.append(train_loss)
    terec.append(test_loss)
    print('Epoch: {:03d}, Train Loss: {:.7f}, Test Loss: {:.7f}'.format(
        epoch, train_loss, test_loss))

    if test_loss < valoss_min*(1 - min_change):
        wait = 0
        valoss_min = test_loss
        best_model_weights = model.state_dict()
    else:
        wait += 1
    if wait > patience:
        break

model.load_state_dict(best_model_weights)

# save the model
torch.save(model.state_dict(), join(outdir, 'model.pt'))

# save the training history
summary = {
    'train_loss': trrec,
    'test_loss': terec,
}
json.dump(summary, open(join(outdir, 'summary.json'), 'w'))


# predict on the test set
def predict_numpy(model, loader):
    true, pred = test(model, loader)
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    true = true*t_std + t_mu
    pred[:, 0] = pred[:, 0]*t_std + t_mu
    pred[:, 1] = pred[:, 1]*(t_std**2)
    return true, pred


true, pred = predict_numpy(model, test_loader)


# save the predictions
np.savez(join(outdir, 'pred.npz'), true=true, pred=pred)

# print the prediction scatter
print(f'Graph-based NLE scatter: {np.std(pred[:,0]-true[:,0]):.4f}')
print(f'Mean prediction scatter: {np.std(np.mean(true)-true[:,0]):.4f}')
