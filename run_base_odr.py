import os
from os.path import join
import numpy as np
import argparse

from sklearn.linear_model import LinearRegression
from scipy.odr import ODR, Model, Data

# ~~~ parse arguments ~~~
parser = argparse.ArgumentParser(
    description="Run SBI inference for toy data.")
parser.add_argument('--data', type=str, default='dC100')
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

dname = f'APR24{args.data}'
vcut = 1e3
Rcut = 1.5
weight = True


def linear_model(B, x):
    return B[0] * x + B[1]


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
print(f'Applying cuts: |vlos| < {vcut} km/s, Rproj < {Rcut} Mpc/h.')


def cut(x):
    x = x[np.abs(x[:, 2]) < vcut]
    Rproj = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    x = x[Rproj < Rcut]
    return x


x_train = [cut(x) for x in x_train]
x_test = [cut(x) for x in x_test]

# ~~~ remove <2 galaxies ~~~
minN = 3
print(f'Removing samples with <{minN} galaxies.')
mask = [len(x) >= minN for x in x_train]
x_train = [x for x, m in zip(x_train, mask) if m]
theta_train = [x for x, m in zip(theta_train, mask) if m]
ids_train = [x for x, m in zip(ids_train, mask) if m]

mask = [len(x) >= minN for x in x_test]
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


# Measure sigmav features
feat_train, feat_test = map(apply, [x_train, x_test])

# Run ODR
data = Data(x=np.array(feat_train).T, y=np.array(theta_train).T)
model = Model(linear_model)
beta0 = [1.0, 12.0]
odr = ODR(data, model, beta0=beta0)
output = odr.run()
coef, intercept = output.beta

# Predict
pred = linear_model([coef, intercept], np.array(feat_test))
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


# Measure richness features
feat_train = np.log10([summ(x[:, 3]) for x in x_train]).reshape(-1, 1)
feat_test = np.log10([summ(x[:, 3]) for x in x_test]).reshape(-1, 1)

# Run ODR
data = Data(x=np.array(feat_train).T, y=np.array(theta_train).T)
model = Model(linear_model)
beta0 = [1.0, 12.0]
odr = ODR(data, model, beta0=beta0)
output = odr.run()
coef, intercept = output.beta
output.pprint()

# Predict
pred = linear_model([coef, intercept], np.array(feat_test))
std = np.std(theta_test - pred)
print('std:', std)

filename = join(outdir, 'Pamico.npz')
out = np.savez(
    filename,
    coef=coef, intercept=intercept, std=std,
    pred=pred, true=theta_test,
    ids=ids_test
)
