# IMPORTS

import os
from os.path import join
import numpy as np
import pickle
from tqdm import tqdm
import seaborn as sns
import pandas as pd

import torch
import sbi
from sbi import utils as utils
from sbi.inference import SNLE, likelihood_estimator_based_potential

import yaml
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams['savefig.dpi'] = 300

# GET CONFIG
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('-v', '--verbose',
                    action='store_true')
args = parser.parse_args()
with open(str(args.config), 'r') as f:
    cfg = dict(yaml.safe_load(f))
print(cfg)
Nclu = cfg['val']['Nclu']
sample_at = cfg['val']['sample_at']
Nsamp = cfg['val']['Nsamp']


# LOAD DATA
datapath = join('data/processed', cfg['data']['name'])
print('Loading from:', datapath)
x = np.load(join(datapath, 'x.npy'))
theta = np.load(join(datapath, 'theta.npy'))
fold = np.load(join(datapath, 'fold.npy'))
ids = np.load(join(datapath, 'id.npy'))

x = x[fold==0]
theta = theta[fold==0]
ids = ids[fold==0]
x = torch.Tensor(x)
theta = torch.Tensor(theta)


# Split by cluster
uids = np.unique(ids)
xs = []
thetas = []
for uid in uids:
    mask = ids==uid
    xs.append(x[mask])
    thetas.append(theta[mask])
Nclu = min(Nclu, len(uids))


# Model path
modelpath = join('saved_models', cfg['data']['name'])
print('Loading from:', modelpath)


# Load trues
trues = np.load(join(modelpath, 'trues.npy'))


# define functions
def plot_scatter(true, pred, title):
    true, pred = np.array(true), np.array(pred)
    f, axs = plt.subplots(1, true.shape[-1], figsize=(5*true.shape[-1], 5))
    for i in range(true.shape[-1]):
        onetone = np.linspace(cfg['priors']['low'][i],cfg['priors']['high'][i],10)
        axs[i].plot(onetone, onetone, 'k--')
        axs[i].plot(true[:,i], pred[:,i], '.')
        axs[i].set_xlabel(cfg['param_names'][i]+'_true')
        axs[i].set_ylabel(cfg['param_names'][i]+'_pred')
        axs[i].set_title(title)
        axs[i].set_xlim(cfg['priors']['low'][i],cfg['priors']['high'][i])
        axs[i].set_ylim(cfg['priors']['low'][i],cfg['priors']['high'][i])
        axs[i].grid()
    f.savefig(join(modelpath, f'{title}.jpg'), dpi=300, bbox_inches='tight')
    for i in range(true.shape[-1]):
        print(f'Correlation {i}:', np.corrcoef(true[:,i], pred[:,i])[1,0])
        
def plot_kde(true, pred, title):
    f, axs = plt.subplots(1, trues.shape[-1], figsize=(5*trues.shape[-1], 5))
    for i in range(trues.shape[-1]):
        x = pd.DataFrame(zip(trues[:,i], samps[:,i]), columns=['true', 'samp'])
        sns.kdeplot(x, fill=True, 
                    x='true', y='samp', ax=axs[i])
        onetone = np.linspace(cfg['priors']['low'][i],cfg['priors']['high'][i],10)
        axs[i].plot(onetone, onetone, 'r--')
        axs[i].set_xlabel(cfg['param_names'][i]+'_true')
        axs[i].set_ylabel(cfg['param_names'][i]+'_pred')
        axs[i].set_title(title)
        axs[i].set_xlim(cfg['priors']['low'][i],cfg['priors']['high'][i])
        axs[i].set_ylim(cfg['priors']['low'][i],cfg['priors']['high'][i])
    f.savefig(join(modelpath, f'{title}.jpg'), dpi=300, bbox_inches='tight')

# Get MAP
for n in sample_at:
    print('n =', n)
    maps = np.load(join(modelpath, f'MAP_n{n}.npy'))
    plot_scatter(trues, maps, 'map_'+cfg['data']['name'] + f'_n={n}')

# Get mean
print('mean_all')
samps = np.load(join(modelpath, 'samps.npy'))
means = np.mean(samps, axis=1)
plot_scatter(trues, means, 'mean_'+cfg['data']['name'] + f'_n=all')


# Get samps
print('samps_all')
trues = np.repeat(np.expand_dims(trues, axis=1), samps.shape[1], axis=1)
trues = trues.reshape(-1, trues.shape[-1])
samps = samps.reshape(-1, samps.shape[-1])
plot_scatter(trues, samps, 'samps_'+cfg['data']['name'] + f'_n=all')
plot_kde(trues, samps, 'sampskde_'+cfg['data']['name'] + f'_n=all')
