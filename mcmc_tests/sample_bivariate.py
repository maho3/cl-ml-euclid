"""Trial: bivariate forward calibration that uses the FULL mass posterior samples
instead of a (mean, std) summary.

For each cluster the mass model gives posterior samples {m_is}. We treat these as
a per-cluster kernel on the true mass and allow an affine calibration correction
(a, b) plus a residual estimator scatter omega0, fit on the labelled subset. The
estimate channel at a candidate true mass m is the sample-averaged

    p(x_i | m) = (1/S) sum_s N(m_is; a + b(m-m0), omega0^2),

which captures the per-cluster width and shape (skew) automatically. Richness is
coupled to the estimate residual through the correlation rho exactly as in the
(mean-only) bivariate model. Setting rho=0 recovers the uncorrelated calibration.

Only used to rerun M4 (rho=0) and M5 (rho free) for comparison; the final
relations and the appendix are untouched.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS

from .data import Data, M0_PIV
from .likelihoods.base import lognorm
from .fit import run_reference


def _model(samps_main, zeta_main, ell_main, mpi_main, sig_pi,
           samps_cal, mtrue_cal, ell_cal, zeta_cal, m_grid, m0, m_ref,
           rho_fixed):
    """rho_fixed: None -> sample rho ~ Uniform (free); a float -> hold rho at
    that value (use 0.0 for the uncorrelated M4, or the calibration-subset
    estimate for the weak channels)."""
    logS_main = jnp.log(samps_main.shape[1])
    logS_cal = jnp.log(samps_cal.shape[1])

    pi0 = numpyro.sample('pi0', dist.Normal(1.4, 0.5))
    Fm = numpyro.sample('Fm', dist.Normal(1.0, 0.5))
    Gz = numpyro.sample('Gz', dist.Normal(0.0, 1.0))
    a = numpyro.sample('a', dist.Normal(m_ref, 0.5))
    b = numpyro.sample('b', dist.TruncatedNormal(0.7, 0.3, low=0.1))
    omega0 = numpyro.sample('omega0', dist.HalfNormal(0.3))
    sigl = numpyro.sample('sigl', dist.HalfNormal(0.3))
    if rho_fixed is None:
        rho = numpyro.sample('rho', dist.Uniform(-0.95, 0.95))
    else:
        rho = numpyro.deterministic('rho', jnp.asarray(float(rho_fixed)))
    c0 = numpyro.sample('c0', dist.Normal(1.4, 0.5))
    c1 = numpyro.sample('c1', dist.Normal(1.0, 0.5))
    c2 = numpyro.sample('c2', dist.Normal(0.0, 1.0))

    beta = rho * sigl / omega0
    sd_cond = jnp.sqrt(sigl ** 2 * (1.0 - rho ** 2))

    # ---- main sample: grid-marginalised, sample-averaged estimate channel ----
    mg = m_grid                                   # (G,)
    cal_mean_g = a + b * (mg - m_ref)             # (G,)
    sm = samps_main[:, None, :]                   # (N,1,S)
    cmg = cal_mean_g[None, :, None]               # (1,G,1)
    ln_est = lognorm(sm, cmg, omega0)             # (N,G,S)  estimate channel
    rel = pi0 + Fm * (mg - m0)[None, :] + Gz * zeta_main[:, None]   # (N,G)
    cross = beta * (sm - cmg)                     # (N,G,S)
    ln_rich = lognorm(ell_main[:, None, None], rel[:, :, None] + cross, sd_cond)
    ln_pi = lognorm(mg[None, :], mpi_main[:, None], sig_pi)         # (N,G)

    ln_est_marg = logsumexp(ln_est, axis=2) - logS_main            # (N,G)
    ln_joint_marg = logsumexp(ln_est + ln_rich, axis=2) - logS_main
    num = logsumexp(ln_pi + ln_joint_marg, axis=1)
    den = logsumexp(ln_pi + ln_est_marg, axis=1)
    numpyro.factor('main', jnp.sum(num - den))

    # ---- calibration subset (true mass known) ----
    dm_cal = mtrue_cal - m_ref
    cal_mean_cal = a + b * dm_cal                 # (Ncal,)
    ln_est_cal = lognorm(samps_cal, cal_mean_cal[:, None], omega0)  # (Ncal,S)
    rel_cal = c0 + c1 * dm_cal + c2 * zeta_cal    # nuisance mean (Ncal,)
    cross_cal = beta * (samps_cal - cal_mean_cal[:, None])
    ln_rich_cal = lognorm(ell_cal[:, None], rel_cal[:, None] + cross_cal, sd_cond)
    ln_joint_cal = logsumexp(ln_est_cal + ln_rich_cal, axis=1) - logS_cal
    numpyro.factor('cal', jnp.sum(ln_joint_cal))


def run_sample_candidate(D, rho_fixed=None, S=40, G=180, settings=None, seed=3,
                         max_tree_depth=7):
    settings = settings or dict(num_warmup=400, num_samples=800, num_chains=2)
    rng = np.random.default_rng(0)
    # subsample posterior samples per cluster
    def sub(arr):
        idx = rng.integers(0, arr.shape[1], size=S)
        return jnp.asarray(arr[:, idx])
    sel, cal = D.sel, D.is_cal
    samps_main = sub(D.samps[sel])
    samps_cal = sub(D.samps[cal])
    m_grid = jnp.asarray(D.mass_grid(G))
    args = (samps_main, jnp.asarray(D.zeta[sel]), jnp.asarray(D.ell[sel]),
            jnp.asarray(D.mpi[sel]), float(D.sig_pi),
            samps_cal, jnp.asarray(D.mtrue[cal]), jnp.asarray(D.ell[cal]),
            jnp.asarray(D.zeta[cal]), m_grid, M0_PIV, M0_PIV, rho_fixed)
    kernel = NUTS(_model, dense_mass=True, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=settings['num_warmup'],
                num_samples=settings['num_samples'],
                num_chains=settings['num_chains'],
                chain_method='vectorized', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), *args)
    return mcmc.get_samples()


# ---------------------------------------------------------------------------
# Driver: full-sample fit + figures, mirroring collab_run.run_bivariate_joint
# (rho fixed at the calibration-subset estimate for consistency with the
# mean-only finals; pass rho_fixed=0.0 for the uncorrelated M4).
# ---------------------------------------------------------------------------
def run_sample_final(model, tag, title, rho_mode='cal', S=30, G=140,
                     settings=None, seed=3, cal_frac=0.1):
    import os
    import json
    from .two_stage import stage1_calibrate
    from .compare import tension
    from .collab_plots import corner_true_pred, scaling_plot, PLOTS
    from .collab_run import RES

    D = Data(model, 'dC100', cal_frac=cal_frac, seed=seed)
    fixed = stage1_calibrate(D)
    if rho_mode == 'cal':
        rho_fixed = float(fixed['rho'])
    elif rho_mode == 'zero':
        rho_fixed = 0.0
    else:
        rho_fixed = None
    cand = run_sample_candidate(D, rho_fixed=rho_fixed, S=S, G=G,
                                settings=settings, seed=seed, max_tree_depth=6)
    ref, _ = run_reference(D, M0_PIV, settings=settings, seed=seed + 1)
    d = tension(cand, ref)['d']
    np.savez(os.path.join(RES, f'{tag}.npz'),
             **{f'cand_{k}': np.asarray(v) for k, v in cand.items()},
             **{f'ref_{k}': np.asarray(v) for k, v in ref.items()})
    json.dump(dict(d=d, rho_cal=fixed['rho'], N_cal=fixed['N_cal']),
              open(os.path.join(RES, f'{tag}.json'), 'w'), indent=2, default=float)
    keys = ['pi0', 'Fm', 'Gz']
    corner_true_pred(cand, ref, keys, os.path.join(PLOTS, f'{tag}_corner.png'),
                     title=f'{title}  (d={d:.2f})')
    scaling_plot(D, cand, ref, os.path.join(PLOTS, f'{tag}_scaling.png'),
                 kind='forward', title=title, step=3)
    print(f'[{tag}] d={d:.2f} Fm={np.median(cand["Fm"]):.3f}/'
          f'{np.median(ref["Fm"]):.3f} rho={rho_fixed}', flush=True)
    return d
