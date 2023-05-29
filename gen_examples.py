# IMPORTS

from os.path import join
import numpy as np
import pickle

import yaml
import argparse

import torch

import matplotlib.pyplot as plt
from corner import corner
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
    Nclu = cfg['example']['Nclu']
    sample_at = cfg['example']['sample_at']
    Nsamp = cfg['example']['Nsamp']

    # LOAD DATA
    datapath = join('data/processed', cfg['data']['name'])
    print('Loading from:', datapath)
    x = np.load(join(datapath, 'x.npy'))
    theta = np.load(join(datapath, 'theta.npy'))
    fold = np.load(join(datapath, 'fold.npy'))
    ids = np.load(join(datapath, 'id.npy'))

    tmin, tmax = theta.min(axis=0), theta.max(axis=0)

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

    def plot_ex(true, samps, data, title):
        f = plt.figure(figsize=(6, 6))
        corner(samps, labels=cfg['param_names'], truths=true,
               quantiles=[0.16, 0.84], plot_contour=False,
               range=[(tmin[i], tmax[i]) for i in range(len(tmin))],
               fig=f)
        ax = f.add_subplot(333)
        ax.axhline(0, color='k', linestyle='--')
        ax.plot(np.sqrt(data[:, 0]**2+data[:, 1]**2), data[:, -1], '.')
        ax.set_xlim(0, 3.8)
        ax.set_ylim(-5000, 5000)
        ax.set_xlabel('Rproj')
        ax.set_ylabel('vlos')
        f.savefig(join(modelpath, f'ex_{title}.jpg'),
                  dpi=300, bbox_inches='tight')

    # Sample Examples
    for i, ind in enumerate(
            np.random.choice(len(xs), size=Nclu, replace=False)):
        for n in sample_at:
            xi = xs[i]
            xi = xi[np.random.choice(
                len(xi), size=int(n*len(xi)), replace=False)]
            ti = thetas[ind]
            posterior = posterior.set_default_x(xi)
            samps = posterior.sample(
                (Nsamp,), show_progress_bars=args.verbose).numpy()

            plot_ex(ti[0], samps, xi, cfg['data']['name'] + f'_n={n}_i{i}')
