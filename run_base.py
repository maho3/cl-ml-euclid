import os
from os.path import join
import numpy as np
import argparse

from sklearn.linear_model import LinearRegression

# ~~~ parse arguments ~~~
parser = argparse.ArgumentParser(
    description="Run SBI inference for toy data.")
parser.add_argument('--data', type=str, default='dC100')
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

dname = f'APR24{args.data}'
vcut = 1e3
weight = True


# ~~~ load data ~~~
datapath = join('data/processed', dname)
print('Loading from:', datapath)
x = np.load(join(datapath, 'x_batch.npy'), allow_pickle=True)
theta = np.load(join(datapath, 'theta_batch.npy'), allow_pickle=True)
folds = np.load(join(datapath, 'folds_batch.npy'), allow_pickle=True)
ids = np.load(join(datapath, 'ids_batch.npy'), allow_pickle=True)

# split
x_train = x[folds != args.fold]
theta_train = theta[folds != args.fold]
ids_train = ids[folds != args.fold]
x_test = x[folds == args.fold]
theta_test = theta[folds == args.fold]
ids_test = ids[folds == args.fold]
# x.shape=[(Ngals, N_feats) for i in Nsamp], [xami, yami, vlos, Pmem]

# outdir
outdir = f'saved_models/apr24_base_{args.data}_f{args.fold}'
print('Saving to:', outdir)
os.makedirs(outdir, exist_ok=True)


# ~~~ apply cuts ~~~
print(f'Applying cuts: vlos < {vcut} km/s.')


def cut(x):
    return x[np.abs(x[:, 2]) < vcut]


x_train = [cut(x) for x in x_train]
x_test = [cut(x) for x in x_test]

# ~~~ remove <2 galaxies ~~~
print('Removing samples with <3 galaxies.')
mask = [len(x) > 2 for x in x_train]
x_train = [x for x, m in zip(x_train, mask) if m]
theta_train = [x for x, m in zip(theta_train, mask) if m]
ids_train = [x for x, m in zip(ids_train, mask) if m]

mask = [len(x) > 2 for x in x_test]
x_test = [x for x, m in zip(x_test, mask) if m]
theta_test = [x for x, m in zip(theta_test, mask) if m]
ids_test = [x for x, m in zip(ids_test, mask) if m]


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
    pred=pred, true=theta_test,
    ids=ids_test
)


# ~~~ PAMICO ~~~
print('Fitting PAMICO.')


def summ(x):
    return np.sum(x)


feat_train = np.log10([summ(x[:, 3]) for x in x_train]).reshape(-1, 1)
feat_test = np.log10([summ(x[:, 3]) for x in x_test]).reshape(-1, 1)

if args.data[0] == 'w':
    fit_min = np.log10(1)
elif args.data[0] == 'd':
    fit_min = np.log10(10)
mask = feat_train[:, 0] > fit_min

lr = LinearRegression().fit(feat_train[mask], np.array(theta_train)[mask])
coef, intercept = lr.coef_, lr.intercept_

pred = lr.predict(feat_test)
# (feat_test - intercept)/coef
std = np.std(theta_test - pred)

filename = join(outdir, 'Pamico.npz')
out = np.savez(
    filename,
    coef=coef, intercept=intercept, std=std,
    pred=pred, true=theta_test,
    ids=ids_test
)
