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


# ~~~ load data ~~~
datapath = join('data/processed', dname)
print('Loading from:', datapath)
theta = np.load(join(datapath, 'theta_batch.npy'), allow_pickle=True)
folds = np.load(join(datapath, 'folds_batch.npy'), allow_pickle=True)
ids = np.load(join(datapath, 'ids_batch.npy'), allow_pickle=True)
metas = np.load(join(datapath, 'metas_batch.npy'),
                allow_pickle=True)  # this has the features

# ~~~ filter ~~~
min_richness = 3
mask = metas[:, 3] >= min_richness
theta = theta[mask]
folds = folds[mask]
ids = ids[mask]
metas = metas[mask]

# split
fold = args.fold
theta_train = theta[folds != fold]
theta_test = theta[folds == fold]
ids_train = ids[folds != fold]
ids_test = ids[folds == fold]
meta_train = metas[folds != fold]
meta_test = metas[folds == fold]

# outdir
outdir = f'saved_models/apr24_base_{args.data}_f{args.fold}'
print('Saving to:', outdir)
os.makedirs(outdir, exist_ok=True)


# ~~~ M-sig ~~~
print('Fitting M-sig.')
feat_train = meta_train[:, 2].reshape(-1, 1)
feat_test = meta_test[:, 2].reshape(-1, 1)
feat_train, feat_test = map(np.log10, [feat_train, feat_test])

lr = LinearRegression().fit(feat_train, theta_train)
coef, intercept = lr.coef_, lr.intercept_

pred = lr.predict(feat_test)
# (feat_test - intercept)/coef
std = np.std(theta_test - pred)
print('std:', std)

filename = join(outdir, 'msig.npz')
out = np.savez(
    filename,
    coef=coef, intercept=intercept, std=std,
    pred=pred, true=theta_test,
    ids=ids_test
)


# ~~~ PAMICO ~~~
print('Fitting spectroscopic richness.')

feat_train = np.log10(meta_train[:, 3]).reshape(-1, 1)
feat_test = np.log10(meta_test[:, 3]).reshape(-1, 1)
feat_train, feat_test = map(np.log10, [feat_train, feat_test])

lr = LinearRegression().fit(feat_train, theta_train)
coef, intercept = lr.coef_, lr.intercept_

pred = lr.predict(feat_test)
# (feat_test - intercept)/coef
std = np.std(theta_test - pred)
print('std:', std)

filename = join(outdir, 'Pamico.npz')
out = np.savez(
    filename,
    coef=coef, intercept=intercept, std=std,
    pred=pred, true=theta_test,
    ids=ids_test
)
