"""Per-likelihood gates: grid-vs-closed regression, inject self-test, speed,
information-free check (CLAUDE.md constraint #5)."""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp as _lse

from .fit import prepare, run_candidate


def _lognorm(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def grid_vs_closed(data, m_ref, m0, p=None):
    """Regression test: for the LINEAR Gaussian-pi sub-case, the conditional
    grid form must equal the closed form to ~1e-8. Returns max abs diff."""
    main = data.main()
    mu_main = np.asarray(main['mu']); sig_main = np.asarray(main['sig'])
    zeta_main = np.asarray(main['zeta']); ell_main = np.asarray(main['ell'])
    mpi_main = np.asarray(main['mpi']); sig_pi = main['sig_pi']
    grid = data.mass_grid()
    if p is None:
        p = dict(pi0=1.4, Fm=1.0, Gz=0.0, sigl=0.2, a=m_ref, b_fwd=0.7,
                 omega0=0.1, kappa=1.0)
    mg = grid[None, :]
    om = np.sqrt(p['omega0'] ** 2 + (p['kappa'] * sig_main) ** 2)[:, None]
    ln_pi = _lognorm(mg, mpi_main[:, None], sig_pi)
    ln_c = _lognorm(mu_main[:, None], p['a'] + p['b_fwd'] * (mg - m_ref), om)
    ln_r = _lognorm(ell_main[:, None],
                    p['pi0'] + p['Fm'] * (mg - m0) + p['Gz'] * zeta_main[:, None],
                    p['sigl'])
    grid_ll = _lse(ln_pi + ln_c + ln_r, axis=1) - _lse(ln_pi + ln_c, axis=1)

    om2 = p['omega0'] ** 2 + (p['kappa'] * sig_main) ** 2
    x = m_ref + (mu_main - p['a']) / p['b_fwd']
    v = om2 / p['b_fwd'] ** 2
    tau2 = 1.0 / (1.0 / v + 1.0 / sig_pi ** 2)
    mhat = tau2 * (x / v + mpi_main / sig_pi ** 2)
    mean_ell = p['pi0'] + p['Fm'] * (mhat - m0) + p['Gz'] * zeta_main
    var_ell = p['sigl'] ** 2 + p['Fm'] ** 2 * tau2
    closed_ll = _lognorm(ell_main, mean_ell, np.sqrt(var_ell))
    return float(np.max(np.abs(grid_ll - closed_ll)))


def speed(data, likelihood, m0, m_ref, n_eval=200):
    """JIT the log-density, time evals + one gradient. Returns ms/eval, ms/grad."""
    import numpyro
    from numpyro.infer.util import potential_energy, log_density
    prepped = prepare(data, likelihood, m_ref)

    def model():
        likelihood.numpyro_model(prepped, m0, m_ref)

    # build a param dict at the prior medians-ish via a quick trace
    from numpyro import handlers
    with handlers.seed(rng_seed=0):
        tr = handlers.trace(model).get_trace()
    params = {k: v['value'] for k, v in tr.items()
              if v['type'] == 'sample' and not v.get('is_observed')}

    def logp(par):
        lp, _ = log_density(model, (), {}, par)
        return lp

    logp_j = jax.jit(logp)
    grad_j = jax.jit(jax.grad(logp))
    logp_j(params).block_until_ready()
    grad_j(params)['pi0'].block_until_ready()
    t0 = time.time()
    for _ in range(n_eval):
        logp_j(params).block_until_ready()
    ms_eval = (time.time() - t0) / n_eval * 1e3
    t0 = time.time()
    for _ in range(n_eval):
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), grad_j(params))
    ms_grad = (time.time() - t0) / n_eval * 1e3
    return ms_eval, ms_grad
