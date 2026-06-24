"""Two-stage (cut) inference, as a comparison to the joint bivariate fit.

Stage 1: estimate the covariance module (a, b, omega, sigl, rho) on the cal
         subset directly from (mu, ell, m_true). Cal richness is used ONLY here.
Stage 2: FREEZE the covariance and fit the relation (pi0, F_m, G_z) on the MAIN
         sample alone (never sees cal richness). rho either hard-frozen or given
         an informative prior N(rho_hat, se) -- the 'soft cut'.

The nuisance c0/c1/c2 of the joint model are unnecessary here: the separation is
by construction.
"""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax.scipy.special import logsumexp

from .data import Data, M0_PIV
from .likelihoods.base import lognorm
from .fit import FULL
from .compare import tension


def stage1_calibrate(D, m_ref=M0_PIV):
    """Estimate covariance module on the cal subset (point estimates + se_rho)."""
    c = D.is_cal
    mu, ell, mt, zeta = D.mu[c], D.ell[c], D.mtrue[c], D.zeta[c]
    dm = mt - m_ref
    # mu -> m map (linear) + omega
    Amu = np.vstack([np.ones_like(dm), dm]).T
    a, b = np.linalg.lstsq(Amu, mu, rcond=None)[0]
    r_mu = mu - (a + b * dm)
    omega = float(np.std(r_mu))
    # richness nuisance relation (intercept + mass + z) -> residual -> sigl, rho
    Ael = np.vstack([np.ones_like(dm), dm, zeta]).T
    coef = np.linalg.lstsq(Ael, ell, rcond=None)[0]
    r_ell = ell - Ael @ coef
    sigl = float(np.std(r_ell))
    rho = float(np.corrcoef(r_mu, r_ell)[0, 1])
    se_rho = float((1 - rho ** 2) / np.sqrt(len(dm)))   # Fisher-ish SE
    return dict(a=float(a), b=float(b), omega=omega, sigl=sigl, rho=rho,
                se_rho=se_rho, N_cal=int(c.sum()))


def stage2_model(mu_main, zeta_main, ell_main, mpi_main, sig_pi, m_grid,
                 fixed, m0, m_ref, soft):
    pi0 = numpyro.sample('pi0', dist.Normal(1.4, 0.5))
    Fm = numpyro.sample('Fm', dist.Normal(1.0, 0.5))
    Gz = numpyro.sample('Gz', dist.Normal(0.0, 1.0))
    if soft:
        rho = numpyro.sample('rho', dist.TruncatedNormal(
            fixed['rho'], fixed['se_rho'], low=-0.95, high=0.95))
    else:
        rho = fixed['rho']
    a, b, omega, sigl = fixed['a'], fixed['b'], fixed['omega'], fixed['sigl']
    beta = rho * sigl / omega
    sd_cond = jnp.sqrt(sigl ** 2 * (1.0 - rho ** 2))

    mg = m_grid[None, :]
    cal_mean_g = a + b * (mg - m_ref)
    ln_pi = lognorm(mg, mpi_main[:, None], sig_pi)
    ln_c = lognorm(mu_main[:, None], cal_mean_g, omega)
    r_mu = mu_main[:, None] - cal_mean_g
    rel = pi0 + Fm * (mg - m0) + Gz * zeta_main[:, None]
    ln_r = lognorm(ell_main[:, None], rel + beta * r_mu, sd_cond)
    num = logsumexp(ln_pi + ln_c + ln_r, axis=1)
    den = logsumexp(ln_pi + ln_c, axis=1)
    numpyro.factor('main', jnp.sum(num - den))


def run_stage2(D, fixed, m0=M0_PIV, m_ref=M0_PIV, soft=False, settings=FULL,
               seed=0):
    main = D.main()
    grid = D.mass_grid()
    args = (jnp.asarray(main['mu']), jnp.asarray(main['zeta']),
            jnp.asarray(main['ell']), jnp.asarray(main['mpi']),
            float(main['sig_pi']), jnp.asarray(grid), fixed, m0, m_ref, soft)
    kernel = NUTS(stage2_model, dense_mass=True)
    mcmc = MCMC(kernel, num_warmup=settings['num_warmup'],
                num_samples=settings['num_samples'],
                num_chains=settings['num_chains'],
                chain_method='vectorized', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), *args)
    return mcmc.get_samples()
