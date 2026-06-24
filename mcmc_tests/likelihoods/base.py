"""Likelihood ABC.

Each likelihood declares which data channels it consumes so we can enforce
hard constraint #2 structurally: the calibration subset's *richness* must never
reach the relation term. A likelihood lists `cal_channels` (allowed: mtrue, mu,
sig) and `main_channels` (allowed: mu, sig, zeta, ell, mpi). 'ell' (richness)
may appear ONLY in main_channels. The driver asserts this.
"""
from abc import ABC, abstractmethod

import jax.numpy as jnp


def lognorm(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - jnp.log(sd) - 0.5 * jnp.log(2 * jnp.pi)


class Likelihood(ABC):
    #: parameter names sampled by the model
    param_names = ()
    #: which cal-subset channels are consumed (subset of {mtrue, mu, sig})
    cal_channels = ()
    #: which main-sample channels are consumed (subset of {mu,sig,zeta,ell,mpi})
    main_channels = ()

    def __init__(self, **cfg):
        self.cfg = cfg

    def assert_constraints(self):
        """Constraint #2: richness ('ell') may only be a MAIN channel, never
        a calibration channel."""
        assert 'ell' not in self.cal_channels, (
            'CONSTRAINT VIOLATION: cal-subset richness reached the calibration '
            'term. The relation must never see cal richness.')
        assert set(self.cal_channels) <= {'mtrue', 'mu', 'sig'}, \
            f'unexpected cal channels: {self.cal_channels}'
        assert set(self.main_channels) <= {'mu', 'sig', 'zeta', 'ell', 'mpi'}, \
            f'unexpected main channels: {self.main_channels}'

    @abstractmethod
    def numpyro_model(self, data, m0, m_ref):
        """Define the numpyro model given a prepared `data` dict."""
        ...
