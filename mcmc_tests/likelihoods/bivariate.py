"""Bivariate forward calibration: joint p(mu, ell | m) with a constant
cross-correlation rho between the mass-estimate channel and the richness
relation.

Motivation (see REPORT.md): gnn_npe's posterior mean mu and the richness ell are
correlated at fixed true mass (corr ~ 0.41), because the graph-net reads the same
galaxy content that drives AMICO richness. The plain forward model factorizes
p(mu|m) p(ell|m) (assumes rho=0) and mis-reads that shared scatter as a steeper,
tighter relation (F_m inflated, sigma_lambda collapsed). This likelihood models
the correlation explicitly.

Model (constant covariance, LINEAR map -- kept simple per the user's request):
    mu  = a + b_fwd*(m - m_ref) + eta_mu
    ell = pi0 + F_m*(m - m0) + G_z*zeta + eta_ell
    (eta_mu, eta_ell) ~ N(0, [[omega^2, rho*omega*sigl],
                              [rho*omega*sigl, sigl^2]])   (rho, omega, sigl const)

Bivariate conditional used in the likelihood:
    ell | mu, m ~ N( relation(m) + beta*(mu - cal_mean(m)),  sigl^2 (1-rho^2) )
    beta = rho*sigl/omega.   beta=0 (rho=0) reduces EXACTLY to the plain model.

Constraint handling (user-authorized relaxation of #2): the calibration subset's
richness is used, but ONLY to calibrate the covariance (rho, sigl). It is
structurally blocked from constraining the headline relation amplitude
(pi0, F_m, G_z) by fitting a throwaway nuisance relation (c0 + c1*dm + c2*zeta)
to the cal-subset richness mean -- so any mean trend of ell with mass/z on the cal
subset is absorbed by the nuisance, and cal richness informs only the residual
covariance. The main-sample term alone constrains (pi0, F_m, G_z).
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp

from .base import Likelihood, lognorm


class BivariateForwardCal(Likelihood):
    cal_channels = ('mtrue', 'mu', 'sig', 'ell', 'zeta')  # ell allowed: cov only
    main_channels = ('mu', 'sig', 'zeta', 'ell', 'mpi')
    uses_cal_richness = True  # authorized: cal ell calibrates rho/sigl only

    def __init__(self, map='linear', rho_map='const', **cfg):
        super().__init__(map=map, rho_map=rho_map, **cfg)
        self.map = map  # 'linear' | 'quadratic' (mu->m mean map)
        self.rho_map = rho_map  # 'const' | 'linear' (rho vs mass)
        names = ['pi0', 'Fm', 'Gz', 'a', 'b_fwd']
        if map == 'quadratic':
            names.append('c_fwd')
        names += ['omega', 'sigl']
        names += ['rho'] if rho_map == 'const' else ['rho0', 'rho1']
        names += ['c0', 'c1', 'c2']
        self.param_names = tuple(names)

    def cal_mean(self, dm, p):
        out = p['a'] + p['b_fwd'] * dm
        if self.map == 'quadratic':
            out = out + p['c_fwd'] * dm ** 2
        return out

    def rho_fn(self, dm, p):
        if self.rho_map == 'const':
            return p['rho']
        return jnp.clip(p['rho0'] + p['rho1'] * dm, -0.95, 0.95)

    def assert_constraints(self):
        # cal richness is intentionally used here, but only via the covariance
        # (nuisance-absorbed mean). Document the relaxation explicitly.
        assert 'ell' in self.cal_channels, 'bivariate uses cal ell for rho'
        assert set(self.main_channels) <= {'mu', 'sig', 'zeta', 'ell', 'mpi'}

    def numpyro_model(self, data, m0, m_ref):
        mu_main = data['mu_main']
        zeta_main = data['zeta_main']
        ell_main = data['ell_main']
        mpi_main = data['mpi_main']
        sig_pi = data['sig_pi']
        mu_cal = data['mu_cal']
        mtrue_cal = data['mtrue_cal']
        ell_cal = data['ell_cal']
        zeta_cal = data['zeta_cal']
        m_grid = data['m_grid']

        pi0 = numpyro.sample('pi0', dist.Normal(1.4, 0.5))
        Fm = numpyro.sample('Fm', dist.Normal(1.0, 0.5))
        Gz = numpyro.sample('Gz', dist.Normal(0.0, 1.0))
        pmap = {}
        pmap['a'] = numpyro.sample('a', dist.Normal(m_ref, 0.5))
        pmap['b_fwd'] = numpyro.sample('b_fwd',
                                       dist.TruncatedNormal(0.7, 0.3, low=0.1))
        if self.map == 'quadratic':
            pmap['c_fwd'] = numpyro.sample('c_fwd', dist.Normal(0.0, 0.5))
        omega = numpyro.sample('omega', dist.HalfNormal(0.3))
        sigl = numpyro.sample('sigl', dist.HalfNormal(0.3))
        if self.rho_map == 'const':
            pmap['rho'] = numpyro.sample('rho', dist.Uniform(-0.95, 0.95))
        else:
            pmap['rho0'] = numpyro.sample('rho0', dist.Uniform(-0.95, 0.95))
            pmap['rho1'] = numpyro.sample('rho1', dist.Normal(0.0, 0.5))
        # throwaway nuisance relation for the cal subset (absorbs the mean so
        # cal richness informs ONLY rho/sigl, never pi0/Fm/Gz)
        c0 = numpyro.sample('c0', dist.Normal(1.4, 0.5))
        c1 = numpyro.sample('c1', dist.Normal(1.0, 0.5))
        c2 = numpyro.sample('c2', dist.Normal(0.0, 1.0))

        def cond_params(dm):
            """beta and conditional sd at mass offset dm (rho may vary)."""
            r = self.rho_fn(dm, pmap)
            return r * sigl / omega, jnp.sqrt(sigl ** 2 * (1.0 - r ** 2))

        # ---- Term 1: calibration subset, covariance only ----
        dm_cal = mtrue_cal - m_ref
        cal_mean_cal = self.cal_mean(dm_cal, pmap)
        r_mu_cal = mu_cal - cal_mean_cal
        beta_cal, sd_cal = cond_params(dm_cal)
        # mu channel -> a, b, omega
        ln_mu = lognorm(mu_cal, cal_mean_cal, omega)
        # ell | mu channel -> rho, sigl (mean fully absorbed by nuisance c0,c1,c2)
        ell_mean_cal = c0 + c1 * dm_cal + c2 * zeta_cal + beta_cal * r_mu_cal
        ln_ell = lognorm(ell_cal, ell_mean_cal, sd_cal)
        numpyro.factor('cal', (ln_mu + ln_ell).sum())

        # ---- Term 2: main sample, conditional grid-marginalized bivariate ----
        mg = m_grid[None, :]
        cal_mean_g = self.cal_mean(mg - m_ref, pmap)
        ln_pi = lognorm(mg, mpi_main[:, None], sig_pi)
        ln_c = lognorm(mu_main[:, None], cal_mean_g, omega)
        r_mu_main = mu_main[:, None] - cal_mean_g
        beta_g, sd_g = cond_params(mg - m_ref)
        rel = pi0 + Fm * (mg - m0) + Gz * zeta_main[:, None]
        ln_r = lognorm(ell_main[:, None], rel + beta_g * r_mu_main, sd_g)
        num = logsumexp(ln_pi + ln_c + ln_r, axis=1)
        den = logsumexp(ln_pi + ln_c, axis=1)
        numpyro.factor('main', jnp.sum(num - den))
