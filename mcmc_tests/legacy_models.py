"""The three 'legacy' models of the COLLAB report, ported from the notebooks so
they run in this framework on the same data / QC / split.

  M1  inverse_v2     -- inverse parametrization p(m | lambda, z); native A,B,C.
  M2  forward_v3     -- forward p(log10 lambda | m, z) with phi(m|z)=p(m) (flat).
  M3  forward_zprior -- forward, but with a z-dependent population prior phi(m|z).

All three marginalise the per-cluster mass posterior by Monte-Carlo over the
mass samples + log-sum-exp (the notebook approach). The 'true-mass' reference is
the same model run with samps = m_true[:, None] (a single exact sample).
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .data import M0_PIV, Z0, SIG0


# ----------------------------------------------------------------------------
# M1: inverse parametrization  (check_mcmc_v2)
#   <m | lambda, z> = A + B*log10(lambda/l0) + C*log10((1+z)/(1+z0))
#   per-sample weight p(m|theta,lambda,z)/p(m),  p(m) ~ N(m0, sig0^2)
# ----------------------------------------------------------------------------
def _inv_mean(lambs, zs, l0, z0, A, B, C):
    return A + B * jnp.log10(lambs / l0) + C * jnp.log10((1 + zs) / (1 + z0))


def _inv_operand(m, lambs, zs, l0, z0, m0, sig0, sigl, A, B, C):
    t1 = (m - m0) ** 2 / (2 * sig0 ** 2)
    mest = _inv_mean(lambs, zs, l0, z0, A, B, C)
    t2 = (m - mest[:, None]) ** 2 / (2 * sigl ** 2)
    return t1 - t2


def model_inverse(samps, lambs, zs, l0, z0, m0, sig0):
    A = numpyro.sample("A", dist.Uniform(5, 20))
    B = numpyro.sample("B", dist.Uniform(-10, 10))
    C = numpyro.sample("C", dist.Uniform(-10, 10))
    sigl = numpyro.sample("sigl", dist.LogUniform(1e-2, 0.6))
    Nclu, Nsamp = samps.shape
    op = _inv_operand(samps, lambs, zs, l0, z0, m0, sig0, sigl, A, B, C)
    ll = -Nclu * jnp.log(sigl) - Nclu * jnp.log(Nsamp) + jnp.sum(
        logsumexp(op, axis=1))
    numpyro.factor("ll", ll)


# ----------------------------------------------------------------------------
# M2: forward, flat phi(m|z)=p(m)  (check_mcmc_v3)
#   <log10 lambda | m, z> = pi0 + Fm*(m - m0) + Gz*log10((1+z)/(1+z0))
#   phi/p cancels -> operand is the richness residual only.
# ----------------------------------------------------------------------------
def _fwd_mean(m, zs, m0, z0, pi0, Fm, Gz):
    return pi0 + Fm * (m - m0) + Gz * jnp.log10((1 + zs) / (1 + z0))


def model_forward(samps, loglambs, zs, m0, z0):
    pi0 = numpyro.sample("pi0", dist.Uniform(-2, 5))
    Fm = numpyro.sample("Fm", dist.Uniform(-10, 10))
    Gz = numpyro.sample("Gz", dist.Uniform(-10, 10))
    sigl = numpyro.sample("sigl", dist.LogUniform(1e-2, 2.0))
    Nclu, Nsamp = samps.shape
    lam_est = _fwd_mean(samps, zs[:, None], m0, z0, pi0, Fm, Gz)
    op = -(loglambs[:, None] - lam_est) ** 2 / (2 * sigl ** 2)
    ll = -Nclu * jnp.log(sigl) - Nclu * jnp.log(Nsamp) + jnp.sum(
        logsumexp(op, axis=1))
    numpyro.factor("ll", ll)


# ----------------------------------------------------------------------------
# M3: forward with z-dependent population prior phi(m|z)=N(mpi(z), sig_pi)
#   importance weight phi(m|z)/p_infer(m), p_infer ~ N(m0_pr, sig0_pr).
# ----------------------------------------------------------------------------
def _lognorm(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - jnp.log(sd)


def model_forward_zprior(samps, loglambs, zs, m0, z0, mpi, sig_pi,
                         m0_pr, sig0_pr):
    pi0 = numpyro.sample("pi0", dist.Uniform(-2, 5))
    Fm = numpyro.sample("Fm", dist.Uniform(-10, 10))
    Gz = numpyro.sample("Gz", dist.Uniform(-10, 10))
    sigl = numpyro.sample("sigl", dist.LogUniform(1e-2, 2.0))
    Nclu, Nsamp = samps.shape
    lam_est = _fwd_mean(samps, zs[:, None], m0, z0, pi0, Fm, Gz)
    rich = -(loglambs[:, None] - lam_est) ** 2 / (2 * sigl ** 2) - jnp.log(sigl)
    # importance reweight: phi(m|z) / p_infer(m)
    logw = (_lognorm(samps, mpi[:, None], sig_pi)
            - _lognorm(samps, m0_pr, sig0_pr))
    op = rich + logw
    ll = -Nclu * jnp.log(Nsamp) + jnp.sum(logsumexp(op, axis=1))
    numpyro.factor("ll", ll)


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def _run(model, args, seed, warmup, samples, chains):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method='vectorized',
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), *args)
    return mcmc.get_samples()


def _pred_samps(D):
    if D.samps is not None:
        return D.samps
    rng = np.random.default_rng(0)
    return D.mu[:, None] + D.sigma[:, None] * rng.standard_normal((D.N, 100))


def fit_legacy(which, D, masses='pred', l0=None, z0=Z0, m0=M0_PIV,
               seed=0, warmup=500, samples=1500, chains=4):
    """Fit one legacy model. `masses`='pred' uses the model posterior samples;
    'true' uses m_true as a single exact sample (the reference)."""
    rs = 10 ** D.ell
    zs = D.z
    loglambs = D.ell
    samps = D.mtrue[:, None] if masses == 'true' else _pred_samps(D)
    samps = jnp.asarray(np.asarray(samps))

    if which == 'inverse':
        l0 = l0 if l0 is not None else float(np.median(rs))
        args = (samps, jnp.asarray(rs), jnp.asarray(zs), l0, z0, m0, SIG0)
        return _run(model_inverse, args, seed, warmup, samples, chains)
    elif which == 'forward':
        args = (samps, jnp.asarray(loglambs), jnp.asarray(zs), m0, z0)
        return _run(model_forward, args, seed, warmup, samples, chains)
    elif which == 'forward_zprior':
        args = (samps, jnp.asarray(loglambs), jnp.asarray(zs), m0, z0,
                jnp.asarray(D.mpi), float(D.sig_pi), m0, SIG0)
        return _run(model_forward_zprior, args, seed, warmup, samples, chains)
    raise ValueError(which)
