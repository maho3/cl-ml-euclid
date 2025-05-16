

from scipy.stats import skewnorm
from scipy.optimize import minimize
import argparse
import pandas as pd
from collections import defaultdict
import numpy as np
from os.path import join
import tqdm
import os
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_likelihood as calc_loglik
import warnings
numpyro.set_host_device_count(5)
warnings.filterwarnings("ignore")

# ~~~ parse arguments ~~~
parser = argparse.ArgumentParser(
    description="Run SBI inference for toy data.")
parser.add_argument('--data', type=str, default='dC100')
parser.add_argument('--model', type=str, default='summ_nle')
args = parser.parse_args()

# Load everything because I'm lazy
datanames = ['wC50', 'wC100', 'dC50', 'dC100']
runnames = ['base', 'gals_nle', 'summ_nle', 'gnn_npe']
modelnames = ['msig', 'pamico', 'mamp', 'gals_nle', 'summ_nle', 'gnn_npe']
Nfolds = 10
folds = np.arange(Nfolds)

# load processed data
header = 'APR24'
datadir = './data/processed'

theta, fold, ids, richs, zclus = {}, {}, {}, {}, {}
for d in datanames:
    dirpath = join(datadir, f'{header}{d}')
    print('Loading:', dirpath)
    theta[d] = np.load(join(dirpath, 'theta_batch.npy'))
    fold[d] = np.load(join(dirpath, 'folds_batch.npy'))
    ids[d] = np.load(join(dirpath, 'ids_batch.npy'))
    # _s = np.load(join(dirpath, 'x_sum.npy'))
    # richs[d] = _s[:,3]  # old richnesses (sum of AMICO+spectra)
    metas = np.load(join(dirpath, 'metas_batch.npy'))
    zclus[d] = metas[:, 3]  # cluster photometric redshift
    richs[d] = metas[:, 1]  # sum of AMICO photometry

Ndata = {d: len(theta[d]) for d in datanames}

# load model predictions
mdir = './saved_models'
Nsamp = 100
preds = defaultdict(dict)
for d in datanames:
    for r in runnames:
        # setup
        if r == 'base':
            preds[d]['msig'] = np.ones((Ndata[d], 2))*np.nan
            preds[d]['pamico'] = np.ones((Ndata[d], 2))*np.nan
        else:
            preds[d][r] = np.empty((Ndata[d], Nsamp, 1))

        # load
        for f in folds:
            if r == 'gnn_npe':
                dirname = f'oct02_{r}_{d}_f{f}'
            else:
                dirname = f'apr24_{r}_{d}_f{f}'
            # dirname = f'apr24_{r}_{d}_f{f}'
            if r == 'base':
                # Msig
                samplefile = join(mdir, dirname, 'msig.npz')
                if not os.path.exists(samplefile):
                    print(f'Skipping {dirname}')
                    continue
                s = np.load(samplefile)
                place_ids = np.searchsorted(ids[d], s['ids'])
                np.put(preds[d]['msig'][:, 0], place_ids, s['pred'])
                np.put(preds[d]['msig'][:, 1], place_ids, s['std'])

                # Pamico
                samplefile = join(mdir, dirname, 'Pamico.npz')
                if not os.path.exists(samplefile):
                    print(f'Skipping {dirname}')
                    continue
                s = np.load(samplefile)
                place_ids = np.searchsorted(ids[d], s['ids'])
                np.put(preds[d]['pamico'][:, 0], place_ids, s['pred'])
                np.put(preds[d]['pamico'][:, 1], place_ids, s['std'])

            else:
                # ML models
                samplefile = join(mdir, dirname, 'posterior_samples.npy')
                if not os.path.exists(samplefile):
                    print(f'Skipping {dirname}')
                    continue
                s = np.load(samplefile)
                s = np.swapaxes(s, 0, 1)
                s = s[:, :Nsamp]  # subsample if necessary
                preds[d][r][fold[d] == f] = s


def r2logm(r):
    # see preprocessing.ipynb for this measurement
    coef = 0.36752
    intercept = -5.30640
    return (np.log10(r)-intercept)/coef

# load mamposst


mamnames = {
    'wC50': 'wide50', 'wC100': 'wide100', 'dC50': 'deep50', 'dC100': 'deep100'
}
modeldir = './saved_models/mamposst_newprior_dec1824'  # mamposst_nov1324/'

for k, v in mamnames.items():
    isamp = pd.read_csv(join(modeldir, f'result_MockFS_NewAMICO_{v}.dat'),
                        delimiter=' ', skipinitialspace=True)
    isamp['id'] = isamp['#ClusterID'].astype(int)
    # convert r200 to logm
    for c in isamp.columns:
        if 'r200' not in c:
            continue
        isamp['logm'+c[4:]] = r2logm(isamp[c])

    # put in preds
    preds[k]['mamp'] = np.ones((Ndata[k], 5))*np.nan
    place_ids = np.searchsorted(ids[k], isamp['id'].values)
    mask = place_ids < Ndata[k]
    _s = isamp[['logmlow(68)', 'logmup(68)', 'logmlow(95)',
                'logmup(95)', 'logmMAM']].values
    preds[k]['mamp'][place_ids[mask]] = _s[mask]

# calculate percentiles from predictions
q = 100*np.array([0.16, 0.84, 0.5, 0.025, 0.975])
percs = defaultdict(dict)
for d in datanames:
    for m in modelnames:
        if m == 'msig' or m == 'pamico':
            t_ = preds[d][m]
            percs[d][m] = np.stack(
                [t_[:, 0]-t_[:, 1], t_[:, 0]+t_[:, 1], t_[:, 0],
                    t_[:, 0]-2*t_[:, 1], t_[:, 0]+2*t_[:, 1]],
                axis=1).T
        elif m == 'mamp':
            if m not in preds[d]:
                continue
            t_ = preds[d][m]
            percs[d][m] = np.stack(
                [t_[:, 0], t_[:, 1], t_[:, 4], t_[:, 2], t_[:, 3]],
                axis=1).T
        else:
            t_ = preds[d][m]
            percs[d][m] = np.percentile(t_, q, axis=1)[..., 0]
# percs is of shape (5, Ndata)
# dim 0 is of order [16, 84, 50, 2.5, 97.5]

# Compute quality control


def quality_control(percs):
    # checks if we have a reasonable median prediction
    # checks if we're not missing a prediction (not nan)
    med = percs[2]
    mask = (med > 12) & (med < 16)

    err = (percs[1] - percs[0])/2
    mask &= err < 1
    return mask


qc = defaultdict(dict)
for d in datanames:
    for m in modelnames:
        qc[d][m] = quality_control(percs[d][m])


# ~~~ NOW LETS GET DOWN TO BUSINESS ~~~
d = args.data
m = args.model
print(f'\n~~~Fitting MCMC calibration for {d} with {m} model.~~~\n')


# Set up model


def mlamb_mean(lambs, zs, l0, z0, A, B, C):
    return A + B*(jnp.log10(lambs/l0)) + C*(jnp.log10((1+zs)/(1+z0)))


def operand(m, lambs, zs, l0, z0, m0, sig0, sigl, A, B, C):
    t1 = (m-m0)**2/(2*(sig0**2))
    mest = mlamb_mean(lambs, zs, l0, z0, A, B, C)
    t2 = (m - mest[:, None])**2/(2*(sigl**2))
    return t1-t2


def log_likelihood(samps, lambs, zs, l0, z0, m0, sig0, sigl, A, B, C):
    to_sum = operand(samps, lambs, zs, l0, z0, m0, sig0, sigl, A, B, C)
    summed = logsumexp(to_sum, axis=1)
    out = -jnp.log(sigl) + jnp.mean(summed)
    return out


def model(samps, lambs, zs, l0, z0, m0, sig0):
    A = numpyro.sample("A", dist.Uniform(5, 20))
    B = numpyro.sample("B", dist.Uniform(-10, 10))
    C = numpyro.sample("C", dist.Uniform(-10, 10))
    # sigl = numpyro.sample("sigl", dist.LogNormal(-1.407, 0.136))
    sigl = numpyro.sample("sigl", dist.Uniform(0, 0.6))
    # sigl = numpyro.sample("sigl", dist.Exponential(4.))

    numpyro.factor(
        "log_likelihood",
        log_likelihood(samps, lambs, zs, l0, z0, m0, sig0, sigl, A, B, C))
    return


def run_mcmc(samps, lambs, zs, l0, z0, m0, sig0,
             warmup=500, samples=1000, chains=4):
    mcmc = MCMC(NUTS(model), num_warmup=warmup,
                num_samples=samples, num_chains=chains)
    mcmc.run(jax.random.PRNGKey(0), samps=samps, lambs=lambs,
             zs=zs, l0=l0, z0=z0, m0=m0, sig0=sig0)
    return mcmc.get_samples()


def fit_skewed_normal(p16, p50, p84):
    target_percentiles = [0.16, 0.50, 0.84]
    observed_values = [p16, p50, p84]

    def objective(params):
        loc, scale, alpha = params
        if scale <= 0:
            return np.inf
        skewed_gaussian = skewnorm(alpha, loc=loc, scale=scale)
        calculated_values = skewed_gaussian.ppf(target_percentiles)
        return np.sum((calculated_values - observed_values) ** 2)

    initial_guess = [p50, (p84 - p16) / 2, 0.5]
    result = minimize(objective, initial_guess)
    loc, scale, alpha = result.x
    return skewnorm(alpha, loc=loc, scale=scale)


# Grab relevant data
# d = 'dC100'
# m = 'gnn_npe'
rs = richs[d]
zs = zclus[d]
ytrue = theta[d][:, 0]
if m in ['msig', 'pamico']:
    mu, sig = preds[d][m].T
    samps = mu[:, None] + sig[:, None]*np.random.randn(len(mu), Nsamp)
elif m == 'mamp':
    ps = preds[d][m]
    p16, p84, p50 = ps[:, 0], ps[:, 1], ps[:, 4]
    samps = []
    for i in tqdm.tqdm(range(len(p16))):
        if np.isnan(p16[i]):
            samps.append(np.nan*np.ones(Nsamp))
            continue
        rvdist = fit_skewed_normal(p16[i], p50[i], p84[i])
        samps.append(rvdist.rvs(Nsamp))
    samps = np.array(samps)
elif m != 'true':
    samps = preds[d][m][..., 0]
else:
    samps = ytrue[:, None]

if m != 'true':
    # quality control
    mask = qc[d][m]
    rs, zs, samps, ytrue = rs[mask], zs[mask], samps[mask], ytrue[mask]

# find pivot
l0, z0 = 15, 1.0  # rs.mean(), zs.mean()

# write down prior
m0, sig0 = 13.78, 0.35

# run MCMC
samples = run_mcmc(samps, rs, zs, l0, z0, m0,
                   sig0, samples=4000, chains=5)

# get log likelihood
loglik = calc_loglik(model, samples, samps, rs, zs, l0,
                     z0, m0, sig0)['log_likelihood']

# save samples
outdir = './saved_samples'
os.makedirs(outdir, exist_ok=True)

# save samples
outpath = join(outdir, f'{d}_{m}_mcmc_samples.npy')
print('\nSaving to:', outpath)
np.save(outpath, samples)

# save log likelihood
outpath = join(outdir, f'{d}_{m}_mcmc_loglik.npy')
print('Saving to:', outpath)
np.save(outpath, loglik)

print('Done!')
