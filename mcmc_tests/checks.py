"""Assumption diagnostics for the forward calibration channel.

Port of the `diagnose` logic from check_calibration_assumptions.ipynb. The
forward channel is mu = a + b*dm + c*dm^2 (+...) + eta, dm = m - m_ref.

`diagnose` returns a dict of numbers and (optionally) saves a 6-panel figure.
`predict_bias` turns those numbers into a qualitative pre-fit prediction of the
sign/magnitude of bias on (pi0, Fm, Gz), per the CLAUDE.md table.
"""
import numpy as np


def binned(x, y, nbins=12, fn=np.std):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    cen, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        if sel.sum() < 5:
            continue
        cen.append(x[sel].mean())
        val.append(fn(y[sel]))
    return np.array(cen), np.array(val)


def diagnose(data, m_ref=13.78, fig_path=None):
    """Run the 6 assumption checks against `data`'s calibration channel
    (which uses data.mu, the chosen summary). Returns metrics dict."""
    mu, sigma, mtrue = data.mu, data.sigma, data.mtrue
    z, skw, zeta = data.z, data.skew, data.zeta
    g = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(mtrue)
    mu, sigma, mtrue, z, skw, zeta = (mu[g], sigma[g], mtrue[g], z[g],
                                      skw[g], zeta[g])

    dm = mtrue - m_ref
    X1 = np.vstack([np.ones_like(dm), dm]).T
    a_, b_ = np.linalg.lstsq(X1, mu, rcond=None)[0]
    resid = mu - (a_ + b_ * dm)
    c0, c1, c2 = np.linalg.lstsq(
        np.vstack([np.ones_like(dm), dm, dm ** 2]).T, mu, rcond=None)[0]
    # cubic curvature term too
    cub = np.linalg.lstsq(
        np.vstack([np.ones_like(dm), dm, dm ** 2, dm ** 3]).T, mu,
        rcond=None)[0]
    zc = np.polyfit(zeta, resid, 1)
    pull = (mtrue - mu) / sigma
    pg = pull[np.isfinite(pull)]
    sg = skw[np.isfinite(skw)]

    metrics = dict(
        N=int(len(mu)), b_fwd=float(b_), quad_c2=float(c2),
        cubic_c3=float(cub[3]), z_slope=float(zc[0]),
        pull_mean=float(pg.mean()), pull_std=float(pg.std()),
        skew_med=float(np.median(sg)),
    )

    if fig_path is not None:
        _plot(mu, sigma, mtrue, z, resid, pull, skw, zeta,
              a_, b_, c0, c1, c2, zc, m_ref, data, fig_path, metrics)
    return metrics


def _plot(mu, sigma, mtrue, z, resid, pull, skw, zeta, a_, b_, c0, c1, c2, zc,
          m_ref, data, fig_path, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    xs = np.linspace(mtrue.min(), mtrue.max(), 100)
    ax[0, 0].plot(mtrue, mu, '.', alpha=0.15, color='C0')
    ax[0, 0].plot(xs, a_ + b_ * (xs - m_ref), 'C3', label=f'linear b={b_:.2f}')
    ax[0, 0].plot(xs, c0 + c1 * (xs - m_ref) + c2 * (xs - m_ref) ** 2, 'C2--',
                  label=f'quad c2={c2:.2f}')
    ax[0, 0].plot(xs, xs, 'k:', label='1:1')
    ax[0, 0].set(xlabel=r'$m_{\rm true}$', ylabel=r'$\mu$', title='(1) forward cal')
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].plot(mtrue, resid, '.', alpha=0.1, color='C0')
    cb, vb = binned(mtrue, resid, fn=np.mean)
    ax[0, 1].plot(cb, vb, 'C3o-'); ax[0, 1].axhline(0, color='k', lw=0.8)
    ax[0, 1].set(xlabel=r'$m_{\rm true}$', ylabel='lin resid',
                 title='(2) nonlinearity')
    ax[0, 2].plot(z, resid, '.', alpha=0.1, color='C0')
    cz, vz = binned(z, resid, fn=np.mean)
    ax[0, 2].plot(cz, vz, 'C3o-'); ax[0, 2].axhline(0, color='k', lw=0.8)
    ax[0, 2].set(xlabel='z', ylabel='lin resid',
                 title=f'(3) z-drift slope={zc[0]:.2f}')
    pgg = pull[np.isfinite(pull)]
    ax[1, 0].hist(pgg, bins=40, density=True, alpha=0.6, color='C0')
    xx = np.linspace(-4, 4, 100)
    ax[1, 0].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), 'C3')
    ax[1, 0].set(xlim=(-4, 4), title=f'(4) pull m={pgg.mean():.2f} s={pgg.std():.2f}')
    cs, vs = binned(sigma, resid, fn=np.std)
    if len(cs):
        ax[1, 1].plot(cs, vs, 'C3o-', label='resid std')
        lim = [min(cs.min(), vs.min()), max(cs.max(), vs.max())]
        ax[1, 1].plot(lim, lim, 'k:', label='omega=sigma')
    ax[1, 1].set(xlabel='reported sigma', ylabel='actual std',
                 title='(5) scatter struct'); ax[1, 1].legend(fontsize=7)
    sgg = skw[np.isfinite(skw)]
    ax[1, 2].hist(sgg, bins=40, alpha=0.6, color='C0')
    ax[1, 2].axvline(0, color='k'); ax[1, 2].axvline(np.median(sgg), color='C3', ls='--')
    ax[1, 2].set(title=f'(6) skew median={np.median(sgg):.2f}')
    fig.suptitle(f'{data.model}/{data.dataset} summary={data.summary} '
                 f'(N={len(mu)})', fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def predict_bias(metrics, map='quadratic'):
    """Qualitative pre-fit prediction given metrics and the map that will be
    used. Returns a human-readable string."""
    lines = []
    c2 = metrics['quad_c2']
    if abs(c2) > 0.05:
        if map == 'linear':
            sign = '+' if c2 > 0 else '-'
            lines.append(f"quad_c2={c2:.2f} != 0 with a LINEAR map -> curved "
                         f"truth gives mass-dependent effective slope, biases "
                         f"F_m ({sign}) and pi0 via pivot.")
        else:
            lines.append(f"quad_c2={c2:.2f} absorbed by {map} map (modeled).")
    if abs(metrics.get('cubic_c3', 0)) > 0.05 and map != 'cubic':
        lines.append(f"cubic_c3={metrics['cubic_c3']:.2f} residual curvature "
                     f"NOT captured by {map} map -> residual F_m bias.")
    if abs(metrics['z_slope']) > 0.05:
        lines.append(f"z_slope={metrics['z_slope']:.2f} -> leaks into G_z "
                     f"unless z-term added.")
    if abs(metrics['pull_mean']) > 0.1:
        lines.append(f"pull_mean={metrics['pull_mean']:.2f} != 0 -> constant "
                     f"offset shifts pi0 (absorbed by free 'a').")
    if abs(metrics['pull_std'] - 1) > 0.1:
        lines.append(f"pull_std={metrics['pull_std']:.2f} != 1 -> width "
                     f"mis-weighting; kappa/omega0 should recalibrate it.")
    if abs(metrics['skew_med']) > 0.1:
        lines.append(f"skew_med={metrics['skew_med']:.2f} != 0 -> Gaussian "
                     f"channel can't undo asymmetric-mean bias; try median/mode "
                     f"summary; if it persists this is OUTSIDE the search space.")
    if not lines:
        lines.append("all checks ~satisfied for this map; expect small bias.")
    return ' '.join(lines)
