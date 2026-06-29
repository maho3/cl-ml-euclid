"""Plotting for the COLLAB report: true-vs-predicted corner overlays and
notebook-style lambda_phot vs M200c scaling plots (format of check_mcmc_v3
cell 21). Works for both the inverse (native A,B,C) and forward (pi0,Fm,Gz)
parametrisations.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

from .data import M0_PIV, Z0

PLOTS = os.path.join(os.path.dirname(__file__), 'plots')

FLABEL = {'pi0': r'$\pi_0$', 'Fm': r'$F_m$', 'Gz': r'$G_z$',
          'sigl': r'$\sigma_\lambda$', 'A': r'$A$', 'B': r'$B$', 'C': r'$C$'}


def corner_true_pred(cand, ref, keys, path, title=''):
    """Overlay predicted (C3) vs true-mass (C0) contours for the given keys."""
    labels = [FLABEL.get(k, k) for k in keys]
    Xr = np.stack([np.asarray(ref[k]) for k in keys], axis=1)
    Xc = np.stack([np.asarray(cand[k]) for k in keys], axis=1)
    rng = [(min(Xr[:, i].min(), Xc[:, i].min()),
            max(Xr[:, i].max(), Xc[:, i].max())) for i in range(len(keys))]
    fig = corner.corner(Xr, labels=labels, color='C0', range=rng,
                        hist_kwargs=dict(density=True), plot_datapoints=False)
    corner.corner(Xc, fig=fig, color='C3', range=rng,
                  hist_kwargs=dict(density=True), plot_datapoints=False)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color='C0', label='true masses'),
                        Line2D([0], [0], color='C3', label='predicted masses')],
               loc='upper right', fontsize=10, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _richness_of_mass(xm, s, kind, m0, z0, zeta=0.0):
    """Draw predicted log10(richness) at masses xm given posterior samples s.
    kind='forward' uses pi0/Fm/Gz/sigl; 'inverse' inverts m=A+B*log10(l)+C*zeta."""
    if kind == 'forward':
        mean = (s['pi0'][None, :] + s['Fm'][None, :] * (xm[:, None] - m0)
                + s['Gz'][None, :] * zeta)
        lines = mean + s['sigl'][None, :] * np.random.randn(*mean.shape)
    else:  # inverse: m = A + B*log10(lambda/l0) + C*zeta  (+ scatter in m)
        # invert for log10(lambda): log10(l/l0) = (m - A - C*zeta)/B
        l0 = s['_l0']
        loglam = (np.log10(l0) + (xm[:, None] - s['A'][None, :]
                  - s['C'][None, :] * zeta) / s['B'][None, :])
        lines = loglam  # scatter here is in mass, not richness; show mean band
    return np.percentile(10 ** lines, [16, 50, 84], axis=1)


def scaling_plot(D, cand, ref, path, kind='forward', title='', l0=None,
                 step=1):
    """lambda_phot (x, log) vs M200c (y): true masses (x), predicted-mass
    posteriors (error bars), and fitted scaling contours (pred + true)."""
    rs = 10 ** D.ell
    ytrue = D.mtrue
    # predicted mass posterior per cluster (for error bars)
    if D.samps is not None:
        p_ = np.percentile(D.samps, [16, 50, 84], axis=1)
    else:
        p_ = np.stack([D.mu - D.sigma, D.mu, D.mu + D.sigma])

    f, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    ax.semilogx()
    sl = slice(None, None, step)
    ax.errorbar(rs[sl], p_[1][sl], yerr=[(p_[1] - p_[0])[sl], (p_[2] - p_[1])[sl]],
                fmt='.', color='C3', alpha=0.5, elinewidth=0.6, ms=3,
                label='predicted mass (posterior)')
    ax.plot(rs[sl], ytrue[sl], 'x', color='C0', ms=4, alpha=0.5,
            label='true mass')

    xm = np.linspace(12.5, 15.3, 100)
    if kind == 'inverse':
        for s in (cand, ref):
            s['_l0'] = l0 if l0 is not None else float(np.median(rs))
    for s, color, lab in [(cand, 'C3', 'fit (predicted masses)'),
                          (ref, 'C0', 'fit (true masses)')]:
        pr = _richness_of_mass(xm, s, kind, M0_PIV, Z0)
        ax.plot(pr[1], xm, c=color, lw=2)
        ax.fill_betweenx(xm, pr[0], pr[2], alpha=0.25, color=color)

    ax.set_xlim(3, 200)
    ax.set_ylim(12.5, 15.3)
    ax.set_xlabel(r'Photometric richness $\lambda_{\rm phot}$')
    ax.set_ylabel(r'$\log_{10}\left[M_{\rm 200c}\ /\ (h^{-1}M_\odot)\right]$')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if title:
        ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    f.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(f)
