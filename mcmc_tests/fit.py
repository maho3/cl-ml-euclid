"""Drivers: prepare the data dict for a likelihood and run MCMC.

The prep step is where hard constraint #2 is enforced at runtime: we build the
cal channel from (mtrue, mu, sig) ONLY and assert no richness array is present.
"""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS

from .likelihoods.forward_cal import model_true

# fast / full settings (CLAUDE.md)
FAST = dict(num_warmup=300, num_samples=800, num_chains=2)
FULL = dict(num_warmup=500, num_samples=2000, num_chains=4)


def prepare(data, likelihood, m_ref, grid_n=301, grid_lo=11.5, grid_hi=16.0):
    """Build the jax data dict consumed by `likelihood.numpyro_model`.

    Enforces constraint #2: the cal dict never contains richness ('ell').
    """
    main = data.main()
    cal = data.cal()
    assert 'ell' not in cal, 'cal channel must not contain richness'
    likelihood.assert_constraints()

    d = dict(
        mu_main=jnp.asarray(main['mu']),
        sig_main=jnp.asarray(main['sig']),
        zeta_main=jnp.asarray(main['zeta']),
        ell_main=jnp.asarray(main['ell']),
        mpi_main=jnp.asarray(main['mpi']),
        sig_pi=float(main['sig_pi']),
        mu_cal=jnp.asarray(cal['mu']),
        sig_cal=jnp.asarray(cal['sig']),
        mtrue_cal=jnp.asarray(cal['mtrue']),
        m_grid=jnp.asarray(data.mass_grid(grid_n, grid_lo, grid_hi)),
    )
    # cal-subset zeta is allowed ONLY for the optional calibration z-term; it is
    # NOT richness, so it does not violate constraint #2.
    if getattr(likelihood, 'z_term', False):
        d['zeta_cal'] = jnp.asarray(data.zeta[data.is_cal])
    # bivariate model (user-authorized): cal richness + zeta, covariance only
    if getattr(likelihood, 'uses_cal_richness', False):
        d['ell_cal'] = jnp.asarray(data.ell[data.is_cal])
        d['zeta_cal'] = jnp.asarray(data.zeta[data.is_cal])
    return d


def run_candidate(data, likelihood, m0, m_ref, settings=None, seed=0,
                  dense_mass=True, max_tree_depth=None):
    settings = settings or FAST
    prepped = prepare(data, likelihood, m_ref)

    def model():
        likelihood.numpyro_model(prepped, m0, m_ref)

    kw = {} if max_tree_depth is None else dict(max_tree_depth=max_tree_depth)
    kernel = NUTS(model, dense_mass=dense_mass, **kw)
    mcmc = MCMC(kernel, num_warmup=settings['num_warmup'],
                num_samples=settings['num_samples'],
                num_chains=settings['num_chains'],
                chain_method='vectorized', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc.get_samples(), mcmc


def run_reference(data, m0, settings=None, seed=1, dense_mass=True):
    """Forward fit on TRUE masses, main sample (constraint: same clusters)."""
    settings = settings or FAST
    ref = data.reference()
    kernel = NUTS(model_true, dense_mass=dense_mass)
    mcmc = MCMC(kernel, num_warmup=settings['num_warmup'],
                num_samples=settings['num_samples'],
                num_chains=settings['num_chains'],
                chain_method='vectorized', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(ref['mtrue']),
             jnp.asarray(ref['zeta']), jnp.asarray(ref['ell']), m0)
    return mcmc.get_samples(), mcmc
