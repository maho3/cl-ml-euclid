"""Data loading for the forward mass-richness calibration.

Ports the loading / QC / per-cluster summary logic from check_mcmc_v7.ipynb and
check_calibration_assumptions.ipynb into a reusable module.

The public entry point is `load_dataset(model, dataset, ...)` which returns a
`Data` object carrying, for the QC'd clusters of the chosen mass model:
  mu, sigma, skew  -- per-cluster posterior summaries (the calibration channel)
  mtrue            -- true M200c (used only by reference fit + cal subset)
  z, zeta          -- redshift and log10((1+z)/(1+z0)) pivot variable
  ell              -- observed log10(richness)  (the relation "data")
  samps            -- (Nclu, Nsamp) raw posterior samples (for median/mode/skew)
and a deterministic, seeded cal/main split.
"""
import os
from os.path import join
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import skewnorm, skew as sstat_skew
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Constants (from the notebooks)
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(join(os.path.dirname(__file__), os.pardir))
MDIR = join(ROOT, 'saved_models')
DATADIR = join(ROOT, 'data', 'processed')
HEADER = 'APR24'

DATANAMES = ['wC50', 'wC100', 'dC50', 'dC100']
RUNNAMES = ['base', 'gals_nle', 'summ_nle', 'gnn_npe']
MODELNAMES = ['msig', 'pamico', 'mamp', 'gals_nle', 'summ_nle', 'gnn_npe']
NFOLDS = 10

# default pivots used throughout
M0_PIV = 13.78
Z0 = 0.82
SIG0 = 0.348


def r2logm(r):
    coef = 0.36752
    intercept = -5.30640
    return (np.log10(r) - intercept) / coef


def fit_skewed_normal(p16, p50, p84):
    target = [0.16, 0.50, 0.84]
    obs = [p16, p50, p84]

    def objective(params):
        loc, scale, alpha = params
        if scale <= 0:
            return np.inf
        sg = skewnorm(alpha, loc=loc, scale=scale)
        return np.sum((sg.ppf(target) - obs) ** 2)

    res = minimize(objective, [p50, (p84 - p16) / 2, 0.5])
    loc, scale, alpha = res.x
    return skewnorm(alpha, loc=loc, scale=scale)


# ---------------------------------------------------------------------------
# Raw loading (cached at module level so repeated calls are cheap)
# ---------------------------------------------------------------------------
_RAW = {}


def _load_raw():
    """Load theta/folds/ids/richness/z/Ngal + per-model predictions for all
    datasets. Mirrors cells 4-9 of check_mcmc_v7.ipynb. Returns a dict."""
    if _RAW:
        return _RAW

    theta, fold, ids, richs, zclus, Ngal = {}, {}, {}, {}, {}, {}
    for d in DATANAMES:
        dirpath = join(DATADIR, f'{HEADER}{d}')
        theta[d] = np.load(join(dirpath, 'theta_batch.npy'))
        fold[d] = np.load(join(dirpath, 'folds_batch.npy'))
        ids[d] = np.load(join(dirpath, 'ids_batch.npy'))
        metas = np.load(join(dirpath, 'metas_batch.npy'))
        zclus[d] = metas[:, 3]
        richs[d] = metas[:, 1]
        Ngal[d] = np.load(join(dirpath, 'x_sum.npy'))[:, -1]
    Ndata = {d: len(theta[d]) for d in DATANAMES}

    Nsamp = 100
    preds = defaultdict(dict)
    for d in DATANAMES:
        for r in RUNNAMES:
            if r == 'base':
                preds[d]['msig'] = np.ones((Ndata[d], 2)) * np.nan
                preds[d]['pamico'] = np.ones((Ndata[d], 2)) * np.nan
            else:
                preds[d][r] = np.full((Ndata[d], Nsamp, 1), np.nan)

            for f in range(NFOLDS):
                dirname = (f'oct02_{r}_{d}_f{f}' if r == 'gnn_npe'
                           else f'apr24_{r}_{d}_f{f}')
                if r == 'base':
                    sf = join(MDIR, dirname, 'msig.npz')
                    if not os.path.exists(sf):
                        continue
                    s = np.load(sf)
                    pid = np.searchsorted(ids[d], s['ids'])
                    np.put(preds[d]['msig'][:, 0], pid, s['pred'])
                    np.put(preds[d]['msig'][:, 1], pid, s['std'])
                    sf = join(MDIR, dirname, 'Pamico.npz')
                    if not os.path.exists(sf):
                        continue
                    s = np.load(sf)
                    pid = np.searchsorted(ids[d], s['ids'])
                    np.put(preds[d]['pamico'][:, 0], pid, s['pred'])
                    np.put(preds[d]['pamico'][:, 1], pid, s['std'])
                else:
                    sf = join(MDIR, dirname, 'posterior_samples.npy')
                    if not os.path.exists(sf):
                        continue
                    s = np.load(sf)
                    s = np.swapaxes(s, 0, 1)[:, :Nsamp]
                    preds[d][r][fold[d] == f] = s

    # mamposst
    mamnames = {'wC50': 'wide50', 'wC100': 'wide100',
                'dC50': 'deep50', 'dC100': 'deep100'}
    modeldir = join(MDIR, 'mamposst_newprior_dec1824')
    for k, v in mamnames.items():
        fpath = join(modeldir, f'result_MockFS_NewAMICO_{v}.dat')
        if not os.path.exists(fpath):
            continue
        isamp = pd.read_csv(fpath, delimiter=' ', skipinitialspace=True)
        isamp['id'] = isamp['#ClusterID'].astype(int)
        for c in isamp.columns:
            if 'r200' not in c:
                continue
            isamp['logm' + c[4:]] = r2logm(isamp[c])
        preds[k]['mamp'] = np.ones((Ndata[k], 5)) * np.nan
        pid = np.searchsorted(ids[k], isamp['id'].values)
        mask = pid < Ndata[k]
        _s = isamp[['logmlow(68)', 'logmup(68)', 'logmlow(95)',
                    'logmup(95)', 'logmMAM']].values
        preds[k]['mamp'][pid[mask]] = _s[mask]

    # percentiles + QC
    q = 100 * np.array([0.16, 0.84, 0.5, 0.025, 0.975])
    percs = defaultdict(dict)
    for d in DATANAMES:
        for m in MODELNAMES:
            if m in ('msig', 'pamico'):
                t_ = preds[d][m]
                percs[d][m] = np.stack(
                    [t_[:, 0] - t_[:, 1], t_[:, 0] + t_[:, 1], t_[:, 0],
                     t_[:, 0] - 2 * t_[:, 1], t_[:, 0] + 2 * t_[:, 1]], axis=1).T
            elif m == 'mamp':
                if m not in preds[d]:
                    continue
                t_ = preds[d][m]
                percs[d][m] = np.stack(
                    [t_[:, 0], t_[:, 1], t_[:, 4], t_[:, 2], t_[:, 3]], axis=1).T
            else:
                t_ = preds[d][m]
                percs[d][m] = np.percentile(t_, q, axis=1)[..., 0]

    qc = defaultdict(dict)
    for d in DATANAMES:
        for m in MODELNAMES:
            if m not in percs[d]:
                continue
            qc[d][m] = _quality_control(percs[d][m])

    # Remove Ngal < 3 (cell 9)
    for d in DATANAMES:
        mask = Ngal[d] >= 3
        theta[d] = theta[d][mask]
        fold[d] = fold[d][mask]
        ids[d] = ids[d][mask]
        richs[d] = richs[d][mask]
        zclus[d] = zclus[d][mask]
        Ngal[d] = Ngal[d][mask]
        Ndata[d] = len(theta[d])
        for m in MODELNAMES:
            if m not in percs[d]:
                continue
            preds[d][m] = preds[d][m][mask]
            percs[d][m] = percs[d][m][:, mask]
            qc[d][m] = qc[d][m][mask]

    _RAW.update(dict(theta=theta, fold=fold, ids=ids, richs=richs,
                     zclus=zclus, Ngal=Ngal, Ndata=Ndata, preds=preds,
                     percs=percs, qc=qc, Nsamp=Nsamp))
    return _RAW


def _quality_control(percs):
    med = percs[2]
    mask = (med > 12) & (med < 16)
    err = (percs[1] - percs[0]) / 2
    mask &= err < 1
    return mask


# ---------------------------------------------------------------------------
# Per-cluster summaries (port of get_summaries) + sample arrays
# ---------------------------------------------------------------------------
def _get_samps(d, m, raw):
    """Return (Nclu, Nsamp) posterior samples for the chosen model, or None for
    Gaussian/analytic channels (msig/pamico)."""
    preds = raw['preds']
    Nsamp = raw['Nsamp']
    if m in ('msig', 'pamico'):
        mu, sig = preds[d][m].T
        return None, mu, sig
    elif m == 'mamp':
        ps = preds[d][m]
        p16, p84, p50 = ps[:, 0], ps[:, 1], ps[:, 4]
        samps = np.array([
            fit_skewed_normal(p16[i], p50[i], p84[i]).rvs(Nsamp)
            if not np.isnan(p16[i]) else np.full(Nsamp, np.nan)
            for i in range(len(p16))])
        return samps, None, None
    else:  # gnn_npe and other ML models
        samps = preds[d][m][..., 0]
        return samps, None, None


# ---------------------------------------------------------------------------
# Public Data container
# ---------------------------------------------------------------------------
class Data:
    """Container with QC'd per-cluster arrays + a seeded cal/main split.

    Calibration channel summary `mu_i` can be 'mean', 'median', or 'mode'.
    Enforces hard constraint #2: the cal-subset richness is dropped entirely so
    it can never enter the relation term.
    """

    def __init__(self, model, dataset='dC100', summary='mean', cal_frac=0.2,
                 seed=0, z0=Z0, m0_piv=M0_PIV):
        raw = _load_raw()
        self.model = model
        self.dataset = dataset
        self.summary = summary
        self.cal_frac = cal_frac
        self.seed = seed
        self.z0 = z0
        self.m0_piv = m0_piv

        d = dataset
        samps, mu_g, sig_g = _get_samps(d, model, raw)
        mtrue_all = raw['theta'][d][:, 0]
        z_all = raw['zclus'][d]
        rs_all = raw['richs'][d]
        qc = raw['qc'][d][model].copy()

        if samps is not None:
            mu_mean = samps.mean(1)
            mu_med = np.median(samps, 1)
            sig = samps.std(1)
            skw = np.asarray(sstat_skew(samps, axis=1, nan_policy='omit'))
        else:  # Gaussian channel
            mu_mean = mu_g
            mu_med = mu_g
            sig = sig_g
            skw = np.zeros_like(mu_g)

        # mask: QC + finite
        finite = np.isfinite(mu_mean) & np.isfinite(sig) & np.isfinite(mtrue_all)
        mask = qc & finite

        self.samps = samps[mask] if samps is not None else None
        self.mu_mean = mu_mean[mask]
        self.mu_med = mu_med[mask]
        self.sigma = sig[mask]
        self.skew = skw[mask]
        self.mtrue = mtrue_all[mask]
        self.z = z_all[mask]
        self.ell = np.log10(rs_all[mask])
        self.zeta = np.log10((1 + self.z) / (1 + z0))

        # choice of posterior summary used as mu_i
        if summary == 'mean':
            self.mu = self.mu_mean
        elif summary == 'median':
            self.mu = self.mu_med
        elif summary == 'mode':
            self.mu = self._mode()
        else:
            raise ValueError(f'unknown summary {summary}')

        self.N = len(self.mu)

        # population prior pi(m|z) = N(mu_phi(z), sig_pi), Gaussian (constraint 3)
        reg_phi = LinearRegression().fit(self.zeta[:, None], self.mtrue)
        self.mpi = reg_phi.predict(self.zeta[:, None])
        self.sig_pi = float(np.std(self.mtrue - self.mpi))
        self.reg_phi = reg_phi

        # deterministic cal/main split
        rng = np.random.default_rng(seed)
        self.is_cal = rng.random(self.N) < cal_frac
        self.sel = ~self.is_cal

    def _mode(self):
        """KDE-free mode estimate: peak of a fine histogram per cluster."""
        if self.samps is None:
            return self.mu_mean
        modes = np.empty(len(self.samps))
        for i, s in enumerate(self.samps):
            s = s[np.isfinite(s)]
            if len(s) < 5:
                modes[i] = np.mean(s) if len(s) else np.nan
                continue
            hist, edges = np.histogram(s, bins=15)
            k = np.argmax(hist)
            modes[i] = 0.5 * (edges[k] + edges[k + 1])
        return modes

    # ---- convenience accessors for the fit (main / cal channels) ----
    def main(self):
        """(mu, sig, zeta, ell, mpi) for the MAIN sample only."""
        s = self.sel
        return dict(mu=self.mu[s], sig=self.sigma[s], zeta=self.zeta[s],
                    ell=self.ell[s], mpi=self.mpi[s], sig_pi=self.sig_pi)

    def cal(self):
        """(mtrue, mu, sig) for the CAL subset only. NO richness (constraint 2)."""
        c = self.is_cal
        return dict(mtrue=self.mtrue[c], mu=self.mu[c], sig=self.sigma[c])

    def reference(self):
        """Full main-sample arrays for the true-mass reference fit."""
        s = self.sel
        return dict(mtrue=self.mtrue[s], zeta=self.zeta[s], ell=self.ell[s])

    def mass_grid(self, n=301, lo=11.5, hi=16.0):
        return np.linspace(lo, hi, n)

    def __repr__(self):
        return (f'<Data {self.model}@{self.dataset} N={self.N} '
                f'(cal={self.is_cal.sum()}, main={self.sel.sum()}) '
                f'summary={self.summary} sig_pi={self.sig_pi:.3f}>')
