# IMPORTS

from os.path import join
import numpy as np
import pickle
from tqdm import tqdm
import math

import torch

import yaml
import argparse

import multiprocessing as mp


if __name__ == '__main__':
    mp.freeze_support()

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

    x = x[fold == 0]
    theta = theta[fold == 0]
    ids = ids[fold == 0]
    x = torch.Tensor(x)
    theta = torch.Tensor(theta)

    # Split by cluster
    uids = np.unique(ids)
    xs = []
    thetas = []
    for uid in uids:
        mask = ids == uid
        xs.append(x[mask])
        thetas.append(theta[mask])
    Nclu = min(Nclu, len(uids))

    # LOAD SNLE
    modelpath = join('saved_models', cfg['data']['name'])
    print('Loading from:', modelpath)

    with open(join(modelpath, 'model.pkl'), 'rb') as handle:
        inference = pickle.load(handle)
    posterior = inference.build_posterior(
        sample_with='mcmc',
        mcmc_method='nuts',
        mcmc_parameters={'num_chains': 5}
    )

    # Save trues
    trues = [x[0].numpy() for x in thetas[:Nclu]]
    np.save(join(modelpath, 'trues.npy'), trues)

    # Get MAP

    # def plot_map(true, pred, title):
    #     true, pred = np.array(true), np.array(pred)
    #     f, axs = plt.subplots(1, true.shape[-1], figsize=(5*true.shape[-1], 5))
    #     for i in range(true.shape[-1]):
    #         onetone = np.linspace(cfg['priors']['low'][i],
    #                               cfg['priors']['high'][i], 10)
    #         axs[i].plot(onetone, onetone, 'k--')
    #         axs[i].plot(true[:, i], pred[:, i], '.')
    #         axs[i].set_xlabel(cfg['param_names'][i]+'_true')
    #         axs[i].set_ylabel(cfg['param_names'][i]+'_pred')
    #         axs[i].set_title(title)
    #     f.savefig(join(modelpath, f'map_{title}.jpg'),
    #               dpi=300, bbox_inches='tight')
    # plot_map(trues, maps, cfg['data']['name'] + f'_n={n}')

    maps = np.zeros((Nclu, len(sample_at), theta.shape[-1]))
    for i in tqdm(range(Nclu), disable=not args.verbose):
        for j, n in enumerate(sample_at):
            xi = xs[i]
            xi = xi[np.random.choice(
                len(xi), size=int(math.ceil(n*len(xi))), replace=False)]
            posterior = posterior.set_default_x(xi)
            maps[i, j] = posterior.map(
                num_iter=200, num_init_samples=100, num_to_optimize=10).numpy()
    np.save(join(modelpath, 'MAP.npy'), maps)

    # Get Samples
    samps = np.zeros((Nclu, Nsamp, theta.shape[-1]))
    for i in tqdm(range(Nclu), disable=not args.verbose):
        xi = xs[i]
        xi = xi[np.random.choice(len(xi), size=int(
            math.ceil(n*len(xi))), replace=False)]
        posterior = posterior.set_default_x(xi)
        samps[i] = posterior.sample((Nsamp,), show_progress_bars=False).numpy()
    np.save(join(modelpath, 'samps.npy'), samps)
