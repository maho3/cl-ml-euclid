import os
from os.path import join
import numpy as np
import argparse

from sklearn.linear_model import LinearRegression

# ~~~ parse arguments ~~~
parser = argparse.ArgumentParser(
    description="Run SBI inference for toy data.")
parser.add_argument('--data', type=str, default='dC100')
args = parser.parse_args()

dname = f'AMICO{args.data}'
vcut = 2e3
weight = True


# ~~~ load data ~~~
datapath = join('data/processed', dname)
print('Loading from:', datapath)
x_train = np.load(join(datapath, 'x_batch_train.npy'), allow_pickle=True)
theta_train = np.load(
    join(datapath, 'theta_batch_train.npy'), allow_pickle=True)[:, 0]
x_test = np.load(join(datapath, 'x_batch_test.npy'), allow_pickle=True)
theta_test = np.load(join(datapath, 'theta_batch_test.npy'),
                     allow_pickle=True)[:, 0]
# x.shape=[(Ngals, N_feats) for i in Nsamp], [xami, yami, vlos, Pmem]

# outdir
outdir = f'saved_models/base_{args.data}'
print('Saving to:', outdir)
os.makedirs(outdir, exist_ok=True)


# ~~~ apply cuts ~~~
print(f'Applying cuts: vlos < {vcut} km/s.')


def cut(x):
    return x[np.abs(x[:, 2]) < vcut]


x_train = [cut(x) for x in x_train]
x_test = [cut(x) for x in x_test]

# ~~~ remove <2 galaxies ~~~
print('Removing samples with <2 galaxies.')
mask = [len(x) > 1 for x in x_train]
x_train = [x for x, m in zip(x_train, mask) if m]
theta_train = [x for x, m in zip(theta_train, mask) if m]

mask = [len(x) > 1 for x in x_test]
x_test = [x for x, m in zip(x_test, mask) if m]
theta_test = [x for x, m in zip(theta_test, mask) if m]


# ~~~ M-sig ~~~
print('Fitting M-sig.')


def summ(x, weights=None):
    if weights is None:
        # gapper
        return np.sqrt(np.sum((x-np.mean(x))**2)/(len(x)-1))
    # weighted by PAMICO
    mu = np.sum(x*weights)/np.sum(weights)
    sig2 = np.sum(weights*(x-mu)**2)/np.sum(weights)
    return np.sqrt(sig2)


def apply(data):
    if weight:
        output = [summ(x[:, 2], x[:, 3]) for x in data]
    else:
        output = [summ(x[:, 2]) for x in data]
    return np.log10(output).reshape(-1, 1)


feat_train, feat_test = map(apply, [x_train, x_test])

lr = LinearRegression().fit(feat_train, theta_train)
coef, intercept = lr.coef_, lr.intercept_

pred = lr.predict(feat_test)
# (feat_test - intercept)/coef
std = np.std(theta_test - pred)

filename = join(outdir, 'msig.npz')
out = np.savez(
    filename,
    coef=coef, intercept=intercept, std=std,
    pred=pred, true=theta_test
)


# ~~~ PAMICO ~~~
print('Fitting PAMICO.')


def summ(x):
    # gapper
    return np.sum(x)


feat_train = np.log10([summ(x[:, 3]) for x in x_train]).reshape(-1, 1)
feat_test = np.log10([summ(x[:, 3]) for x in x_test]).reshape(-1, 1)

lr = LinearRegression().fit(feat_train, theta_train)
coef, intercept = lr.coef_, lr.intercept_

pred = lr.predict(feat_test)
# (feat_test - intercept)/coef
std = np.std(theta_test - pred)

filename = join(outdir, 'Pamico.npz')
out = np.savez(
    filename,
    coef=coef, intercept=intercept, std=std,
    pred=pred, true=theta_test
)
