"""Compare candidate vs reference fits in the 3D theta=(pi0,Fm,Gz) space.

Success: Mahalanobis distance d = sqrt(dmu^T (Sig_ref+Sig_cand)^-1 dmu) <= 1.
sigma_lambda is excluded (CLAUDE.md).
"""
import json

import numpy as np

THETA = ['pi0', 'Fm', 'Gz']


def _mat(samples):
    X = np.stack([np.asarray(samples[k]) for k in THETA], axis=1)
    return X.mean(0), np.cov(X, rowvar=False)


def tension(cand, ref):
    """Return dict with mu_cand, mu_ref, d, and PASS flag."""
    mu_c, S_c = _mat(cand)
    mu_r, S_r = _mat(ref)
    S = S_c + S_r
    dmu = mu_c - mu_r
    d = float(np.sqrt(dmu @ np.linalg.solve(S, dmu)))
    return dict(
        theta=THETA,
        mu_cand=mu_c.tolist(), mu_ref=mu_r.tolist(),
        Sigma_cand=S_c.tolist(), Sigma_ref=S_r.tolist(),
        d=d, PASS=bool(d <= 1.0),
    )


def save_compare(result, path):
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)


def corner_overlay(cand, ref, path, title=''):
    import corner
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [r'$\pi_0$', r'$F_m$', r'$G_z$']
    Xr = np.stack([np.asarray(ref[k]) for k in THETA], axis=1)
    Xc = np.stack([np.asarray(cand[k]) for k in THETA], axis=1)
    fig = corner.corner(Xr, labels=labels, color='C0',
                        hist_kwargs=dict(density=True), plot_datapoints=False)
    corner.corner(Xc, fig=fig, color='C3',
                  hist_kwargs=dict(density=True), plot_datapoints=False)
    fig.suptitle(title)
    # legend proxies
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color='C0', label='reference (true mass)'),
                        Line2D([0], [0], color='C3', label='candidate')],
               loc='upper right', fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
