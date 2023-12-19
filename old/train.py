# IMPORTS

import os
from os.path import join
import numpy as np
import pickle

import torch
from sbi import utils as utils
from sbi.inference import SNLE

import yaml
import argparse


# GET CONFIG
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
print(args)
with open(str(args.config), 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg)


# LOAD DATA
datapath = join('data/processed', cfg['data']['name'])
print('Loading from:', datapath)
x = np.load(join(datapath, 'x.npy'))
theta = np.load(join(datapath, 'theta.npy'))
fold = np.load(join(datapath, 'fold.npy'))

x = x[fold != 0]
theta = theta[fold != 0]
x = torch.Tensor(x)
theta = torch.Tensor(theta)


# TRAIN SNLE
print('Training...')
prior_lims = (cfg['priors']['low'], cfg['priors']['high'])
prior = utils.BoxUniform(low=torch.Tensor(prior_lims[0]),
                         high=torch.Tensor(prior_lims[1]), device='cpu')
inference = SNLE(prior, density_estimator='maf', device='cpu')
inference = inference.append_simulations(theta, x)
inference.train(show_train_summary=True)


# SAVE
modelpath = join('saved_models', cfg['data']['name'])
print('Saving to:', modelpath)
os.makedirs(modelpath, exist_ok=True)

with open(join(modelpath, 'model.pkl'), 'wb') as handle:
    pickle.dump(inference, handle)
