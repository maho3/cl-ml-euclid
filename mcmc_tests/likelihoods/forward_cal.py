"""Forward-calibration, grid-marginalized mass-richness likelihood.

Port of `model_v7` from check_mcmc_v7.ipynb, generalized so the search space of
hard constraint #3 (calibration map + width/scatter model) is exposed as config:

  map: 'linear' | 'quadratic' | 'cubic'   (+ optional z-term)
        mu = a + b*dm + c*dm^2 + e*dm^3 (+ g_z*zeta) + eta,   dm = m - m_ref
  z_term: bool   -- add a linear zeta term to the calibration mean
  width: 'v7'    -- omega^2 = omega0^2 + (kappa*sigma)^2   (default)
         'floor' -- adds a free additive floor (== omega0 already does this;
                    kept for naming parity) -- here 'v7' already has the floor.

The relation term is the conditional grid form (two log-sum-exps) so p(mu_i)
divides out -- this is what protects against the F_m-sigma_lambda degeneracy.

Forward direction (estimate-on-truth): models p(mu | m). Headline-legal.
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp, log_ndtr

from .base import Likelihood, lognorm


def relation_mean(m, zeta, m0, pi0, Fm, Gz):
    return pi0 + Fm * (m - m0) + Gz * zeta


def logskewnorm(x, xi, omega, alpha):
    """Skew-normal log-pdf with location xi, scale omega, shape alpha.
    f = (2/omega) phi(z) Phi(alpha z),  z=(x-xi)/omega.  alpha=0 -> Gaussian.
    DIAGNOSTIC ONLY: a non-Gaussian forward channel is outside the constraint-#3
    headline search space (kept for comparison, per CLAUDE.md)."""
    z = (x - xi) / omega
    return (jnp.log(2.0) - jnp.log(omega) + lognorm(z, 0.0, 1.0)
            + log_ndtr(alpha * z))


class ForwardCal(Likelihood):
    param_names = ('pi0', 'Fm', 'Gz', 'sigl', 'a', 'b_fwd',
                   'omega0', 'kappa')
    cal_channels = ('mtrue', 'mu', 'sig')
    main_channels = ('mu', 'sig', 'zeta', 'ell', 'mpi')

    def __init__(self, map='quadratic', z_term=False, width='v7',
                 channel='gaussian', **cfg):
        super().__init__(map=map, z_term=z_term, width=width, channel=channel,
                         **cfg)
        self.map = map
        self.z_term = z_term
        self.width = width  # 'v7' | 'hetero' (mass-dependent forward scatter)
        # channel: 'gaussian' (headline) | 'skewnorm' (DIAGNOSTIC, non-Gaussian)
        self.channel = channel
        names = ['pi0', 'Fm', 'Gz', 'sigl', 'a', 'b_fwd']
        if map in ('quadratic', 'cubic'):
            names.append('c_fwd')
        if map == 'cubic':
            names.append('e_fwd')
        if z_term:
            names.append('gz_cal')
        names += ['omega0', 'kappa']
        if width == 'hetero':
            names.append('xi')
        if channel == 'skewnorm':
            names += ['alpha0', 'alpha1']  # shape, linear in (m - m_ref)
        self.param_names = tuple(names)

    def channel_logpdf(self, x, m, omega, p, m_ref, zeta=None):
        """log p(mu=x | m): Gaussian (location = cal_mean) or skew-normal
        (location = cal_mean, shape alpha(m) = alpha0 + alpha1*(m-m_ref))."""
        loc = self.cal_mean_fn(m, m_ref, p, zeta)
        if self.channel == 'skewnorm':
            alpha = p['alpha0'] + p['alpha1'] * (m - m_ref)
            return logskewnorm(x, loc, omega, alpha)
        return lognorm(x, loc, omega)

    def omega_fn(self, sigma, m, m_ref, omega0, kappa, xi=None):
        """Forward-channel scatter omega(m, sigma). 'hetero' adds a mass-
        dependent term so omega grows with |m - m_ref| (the measured scatter
        increases at high mass), broadening the latent-mass marginal there."""
        var = omega0 ** 2 + (kappa * sigma) ** 2
        if self.width == 'hetero' and xi is not None:
            var = var + (xi * (m - m_ref)) ** 2
        return jnp.sqrt(var)

    def cal_mean_fn(self, m, m_ref, p, zeta=None):
        dm = m - m_ref
        out = p['a'] + p['b_fwd'] * dm
        if self.map in ('quadratic', 'cubic'):
            out = out + p['c_fwd'] * dm ** 2
        if self.map == 'cubic':
            out = out + p['e_fwd'] * dm ** 3
        if self.z_term and zeta is not None:
            out = out + p['gz_cal'] * zeta
        return out

    def numpyro_model(self, data, m0, m_ref):
        mu_main = data['mu_main']
        sig_main = data['sig_main']
        zeta_main = data['zeta_main']
        ell_main = data['ell_main']
        mpi_main = data['mpi_main']
        sig_pi = data['sig_pi']
        mu_cal = data['mu_cal']
        sig_cal = data['sig_cal']
        mtrue_cal = data['mtrue_cal']
        m_grid = data['m_grid']
        # cal subset zeta only used by the calibration map's optional z-term
        zeta_cal = data.get('zeta_cal', None)

        p = {}
        p['pi0'] = numpyro.sample('pi0', dist.Normal(1.4, 0.5))
        p['Fm'] = numpyro.sample('Fm', dist.Normal(1.0, 0.5))
        p['Gz'] = numpyro.sample('Gz', dist.Normal(0.0, 1.0))
        sigl = numpyro.sample('sigl', dist.HalfNormal(0.3))
        p['a'] = numpyro.sample('a', dist.Normal(m_ref, 0.5))
        p['b_fwd'] = numpyro.sample('b_fwd',
                                    dist.TruncatedNormal(0.7, 0.3, low=0.1))
        if self.map in ('quadratic', 'cubic'):
            p['c_fwd'] = numpyro.sample('c_fwd', dist.Normal(0.0, 0.5))
        if self.map == 'cubic':
            p['e_fwd'] = numpyro.sample('e_fwd', dist.Normal(0.0, 0.5))
        if self.z_term:
            p['gz_cal'] = numpyro.sample('gz_cal', dist.Normal(0.0, 0.5))
        omega0 = numpyro.sample('omega0', dist.HalfNormal(0.2))
        kappa = numpyro.sample('kappa', dist.LogNormal(0.0, 0.3))
        xi = None
        if self.width == 'hetero':
            xi = numpyro.sample('xi', dist.HalfNormal(0.3))
        if self.channel == 'skewnorm':
            p['alpha0'] = numpyro.sample('alpha0', dist.Normal(0.0, 5.0))
            p['alpha1'] = numpyro.sample('alpha1', dist.Normal(0.0, 5.0))

        # ---- Term 1: calibration on the true-mass subset ----
        omega_cal = self.omega_fn(sig_cal, mtrue_cal, m_ref, omega0, kappa, xi)
        numpyro.factor('cal', self.channel_logpdf(
            mu_cal, mtrue_cal, omega_cal, p, m_ref, zeta_cal).sum())

        # ---- Term 2: main sample, conditional grid-marginalized form ----
        mg = m_grid[None, :]
        omega_main = self.omega_fn(sig_main[:, None], mg, m_ref,
                                   omega0, kappa, xi)
        ln_pi = lognorm(mg, mpi_main[:, None], sig_pi)
        ln_c = self.channel_logpdf(
            mu_main[:, None], mg, omega_main, p, m_ref,
            zeta_main[:, None] if self.z_term else None)
        ln_r = lognorm(ell_main[:, None],
                       relation_mean(mg, zeta_main[:, None], m0,
                                     p['pi0'], p['Fm'], p['Gz']), sigl)
        num = logsumexp(ln_pi + ln_c + ln_r, axis=1)
        den = logsumexp(ln_pi + ln_c, axis=1)
        numpyro.factor('main', jnp.sum(num - den))


def model_true(mtrue, zeta, loglam, m0):
    """Reference: relation fit directly from true masses."""
    pi0 = numpyro.sample('pi0', dist.Normal(1.4, 0.5))
    Fm = numpyro.sample('Fm', dist.Normal(1.0, 0.5))
    Gz = numpyro.sample('Gz', dist.Normal(0.0, 1.0))
    sigl = numpyro.sample('sigl', dist.HalfNormal(0.3))
    mean_lam = relation_mean(mtrue, zeta, m0, pi0, Fm, Gz)
    numpyro.factor('rel', dist.Normal(mean_lam, sigl).log_prob(loglam).sum())
