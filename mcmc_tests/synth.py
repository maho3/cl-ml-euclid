"""Synthetic Data for constraint #5 gates: inject self-test and info-free check.

Builds a duck-typed object exposing the same interface fit.prepare/run_* use.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

from .data import M0_PIV, Z0


class SynthData:
    def __init__(self, mu, sigma, mtrue, zeta, ell, mpi, sig_pi, is_cal,
                 model='synth', dataset='synth', summary='mean'):
        self.mu = mu; self.sigma = sigma; self.mtrue = mtrue
        self.zeta = zeta; self.ell = ell; self.mpi = mpi
        self.sig_pi = sig_pi; self.is_cal = is_cal; self.sel = ~is_cal
        self.N = len(mu); self.model = model; self.dataset = dataset
        self.summary = summary; self.z = (10 ** zeta) * (1 + Z0) - 1
        self.skew = np.zeros_like(mu)

    def main(self):
        s = self.sel
        return dict(mu=self.mu[s], sig=self.sigma[s], zeta=self.zeta[s],
                    ell=self.ell[s], mpi=self.mpi[s], sig_pi=self.sig_pi)

    def cal(self):
        c = self.is_cal
        return dict(mtrue=self.mtrue[c], mu=self.mu[c], sig=self.sigma[c])

    def reference(self):
        s = self.sel
        return dict(mtrue=self.mtrue[s], zeta=self.zeta[s], ell=self.ell[s])

    def mass_grid(self, n=301, lo=11.5, hi=16.0):
        return np.linspace(lo, hi, n)


def _truth(real):
    """Fit the true-mass relation on `real` to get plausible injection truth."""
    dm = real.mtrue - M0_PIV
    X = np.stack([np.ones_like(dm), dm, real.zeta], axis=1)
    coef = np.linalg.lstsq(X, real.ell, rcond=None)[0]
    pi0, Fm, Gz = coef
    sigl = float(np.std(real.ell - X @ coef))
    return pi0, Fm, Gz, sigl


def make_inject(real, seed=0, a_star=13.78, b_star=0.7, c_star=0.2,
                omega0_star=0.1, kappa_star=1.0):
    """Forward-inject mu = a + b*dm + c*dm^2 + eta from real masses + real ell.
    The relation truth = the true-mass fit on `real`."""
    rng = np.random.default_rng(seed)
    m_ref = M0_PIV
    mtrue = real.mtrue.copy()
    zeta = real.zeta.copy()
    ell = real.ell.copy()
    sigma = rng.uniform(0.15, 0.35, len(mtrue))
    dm = mtrue - m_ref
    eta_sd = np.sqrt(omega0_star ** 2 + (kappa_star * sigma) ** 2)
    mu = a_star + b_star * dm + c_star * dm ** 2 + eta_sd * rng.standard_normal(len(mtrue))
    reg = LinearRegression().fit(zeta[:, None], mtrue)
    mpi = reg.predict(zeta[:, None]); sig_pi = float(np.std(mtrue - mpi))
    return SynthData(mu, sigma, mtrue, zeta, ell, mpi, sig_pi, real.is_cal.copy(),
                     model='inject'), dict(a=a_star, b_fwd=b_star, c_fwd=c_star,
                                           omega0=omega0_star, kappa=kappa_star)


def make_skew_inject(real, seed=0, a_star=13.78, b_star=0.31, c_star=0.30,
                     omega0_star=0.11, kappa_star=0.39, alpha0=4.0, alpha1=-6.0):
    """Inject a SKEW-NORMAL forward channel with mass-dependent shape
    alpha(m)=alpha0+alpha1*dm (mimics gnn's +1.2->0 residual skew). Used to
    validate that the skew-normal likelihood recovers F_m where Gaussian can't."""
    from scipy.stats import skewnorm
    rng = np.random.default_rng(seed)
    m_ref = M0_PIV
    mtrue = real.mtrue.copy(); zeta = real.zeta.copy(); ell = real.ell.copy()
    sigma = rng.uniform(0.20, 0.35, len(mtrue))
    dm = mtrue - m_ref
    loc = a_star + b_star * dm + c_star * dm ** 2
    omega = np.sqrt(omega0_star ** 2 + (kappa_star * sigma) ** 2)
    alpha = alpha0 + alpha1 * dm
    mu = np.array([skewnorm.rvs(alpha[i], loc=loc[i], scale=omega[i],
                                random_state=rng) for i in range(len(dm))])
    reg = LinearRegression().fit(zeta[:, None], mtrue)
    mpi = reg.predict(zeta[:, None]); sig_pi = float(np.std(mtrue - mpi))
    return SynthData(mu, sigma, mtrue, zeta, ell, mpi, sig_pi,
                     real.is_cal.copy(), model='skewinject')


def make_corr_inject(real, seed=0, rho=0.41, a_star=13.78, b_star=0.31,
                     c_star=0.30, sd_mu=0.155, fm=0.334, pi0=1.250, gz=2.075,
                     sd_ell=0.134):
    """Inject mu AND ell from m_true with CORRELATED residuals (corr=rho),
    breaking the forward model's conditional-independence assumption. Tests
    whether shared mu/richness info reproduces the real F_m inflation."""
    rng = np.random.default_rng(seed)
    m_ref = M0_PIV
    mtrue = real.mtrue.copy(); zeta = real.zeta.copy()
    dm = mtrue - m_ref
    cov = np.array([[sd_mu ** 2, rho * sd_mu * sd_ell],
                    [rho * sd_mu * sd_ell, sd_ell ** 2]])
    eta = rng.multivariate_normal([0, 0], cov, size=len(dm))
    mu = a_star + b_star * dm + c_star * dm ** 2 + eta[:, 0]
    ell = pi0 + fm * dm + gz * zeta + eta[:, 1]
    sigma = rng.uniform(0.20, 0.35, len(dm))
    reg = LinearRegression().fit(zeta[:, None], mtrue)
    mpi = reg.predict(zeta[:, None]); sig_pi = float(np.std(mtrue - mpi))
    return SynthData(mu, sigma, mtrue, zeta, ell, mpi, sig_pi,
                     real.is_cal.copy(), model='corrinject'), \
        dict(pi0=pi0, Fm=fm, Gz=gz)


def make_infofree(real, seed=0):
    """Mass 'posteriors' carry ZERO per-cluster info: mu drawn from pi(m|z),
    sigma large. A correctly-built likelihood should be UNCONSTRAINED in F_m."""
    rng = np.random.default_rng(seed)
    mtrue = real.mtrue.copy(); zeta = real.zeta.copy(); ell = real.ell.copy()
    reg = LinearRegression().fit(zeta[:, None], mtrue)
    mpi = reg.predict(zeta[:, None]); sig_pi = float(np.std(mtrue - mpi))
    # mu independent of mtrue: drawn from the population prior
    mu = mpi + sig_pi * rng.standard_normal(len(mtrue))
    sigma = np.full(len(mtrue), sig_pi)  # as wide as the prior -> no info
    return SynthData(mu, sigma, mtrue, zeta, ell, mpi, sig_pi, real.is_cal.copy(),
                     model='infofree')
