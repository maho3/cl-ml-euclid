"""Final-plot helper: full-parameter corner (candidate, with true-mass reference
medians marked) and the calibrated p(mass | lambda) from the fitted relation.

Usage: python -m mcmc_tests.make_final_plots <experiment_name> [--seed 3]
"""
import argparse
import json
from os.path import join, dirname

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

from .data import Data, M0_PIV, Z0

EXPDIR = join(dirname(__file__), 'experiments')

LABELS = {
    'pi0': r'$\pi_0$', 'Fm': r'$F_m$', 'Gz': r'$G_z$', 'a': r'$a$',
    'b_fwd': r'$b_{\rm fwd}$', 'c_fwd': r'$c_{\rm fwd}$', 'omega': r'$\omega$',
    'sigl': r'$\sigma_\lambda$', 'rho': r'$\rho$', 'c0': r'$c_0$',
    'c1': r'$c_1$', 'c2': r'$c_2$',
}
PRIMARY = ['pi0', 'Fm', 'Gz']


def full_corner(name):
    d = np.load(join(EXPDIR, name, 'samples.npz'))
    comp = json.load(open(join(EXPDIR, name, 'compare.json')))
    cand_keys = [k[5:] for k in d.files if k.startswith('cand_')]
    # order: primary first, then the rest
    order = PRIMARY + [k for k in cand_keys if k not in PRIMARY]
    X = np.stack([d['cand_' + k] for k in order], axis=1)
    labels = [LABELS.get(k, k) for k in order]
    # reference medians as truth lines (only on shared params)
    truths = [float(np.median(d['ref_' + k])) if 'ref_' + k in d.files else None
              for k in order]
    fig = corner.corner(X, labels=labels, truths=truths, truth_color='C0',
                        color='C3', hist_kwargs=dict(density=True),
                        plot_datapoints=False, label_kwargs=dict(fontsize=9))
    fig.suptitle(f'{name}  —  full posterior (red=candidate, blue lines=true-mass '
                 f'reference)\nd(π₀,F_m,G_z) = {comp["d"]:.3f} '
                 f'({"PASS" if comp["PASS"] else "FAIL"})', fontsize=11)
    out = join(EXPDIR, name, 'corner_full.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
    return out


def p_mass_given_lambda(name, model='gnn_npe', data='dC100', cal_frac=0.5,
                        seed=3):
    """p(m | lambda, z) from the fitted relation + population prior pi(m|z).
    Linear-Gaussian => analytic Gaussian posterior. Richness-only inversion
    (no GNN mu); with mu you'd condition on both and get tighter posteriors."""
    d = np.load(join(EXPDIR, name, 'samples.npz'))
    p = {k: float(np.median(d['cand_' + k])) for k in
         ['pi0', 'Fm', 'Gz', 'sigl']}
    D = Data(model, data, cal_frac=cal_frac, seed=seed)
    sig_pi = D.sig_pi
    zeta0 = float(np.median(D.zeta))            # representative redshift
    z0_repr = (10 ** zeta0) * (1 + Z0) - 1
    mpi0 = float(D.reg_phi.predict([[zeta0]])[0])   # prior mean at that z

    m0 = M0_PIV
    A = p['pi0'] - p['Fm'] * m0 + p['Gz'] * zeta0   # ell = A + Fm*m + N(0,sigl)
    # posterior m | ell : precision-weighted
    inv_var = 1.0 / sig_pi ** 2 + p['Fm'] ** 2 / p['sigl'] ** 2
    s_post = np.sqrt(1.0 / inv_var)

    def post(ell):
        mean = s_post ** 2 * (mpi0 / sig_pi ** 2
                              + p['Fm'] * (ell - A) / p['sigl'] ** 2)
        return mean, s_post

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    # (1) posterior densities for a few observed richnesses
    lam_list = [20, 35, 60, 100]
    mgrid = np.linspace(12.8, 15.0, 400)
    for lam in lam_list:
        ell = np.log10(lam)
        mean, sd = post(ell)
        pdf = np.exp(-0.5 * ((mgrid - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax[0].plot(mgrid, pdf, label=fr'$\lambda={lam}$: '
                   fr'$\hat m={mean:.2f}\pm{sd:.2f}$')
    ax[0].set(xlabel=r'$\log_{10} M_{200c}$', ylabel=r'$p(m\,|\,\lambda, z)$',
              title=f'(1) calibrated mass posterior  (z≈{z0_repr:.2f})')
    ax[0].legend(fontsize=8)
    # (2) m_hat(lambda) band over a richness range
    lams = np.logspace(np.log10(10), np.log10(200), 100)
    means = np.array([post(np.log10(l))[0] for l in lams])
    ax[1].plot(lams, means, 'C3', label=r'$\hat m(\lambda)$ (posterior mean)')
    ax[1].fill_between(lams, means - s_post, means + s_post, color='C3',
                       alpha=0.25, label=r'$\pm 1\sigma$')
    # mean forward relation (for reference): m s.t. ell = A + Fm*m  -> invert
    m_fwd = (np.log10(lams) - A) / p['Fm']
    ax[1].plot(lams, m_fwd, 'k:', label='forward mean (no prior/Eddington)')
    ax[1].set(xscale='log', xlabel=r'$\lambda$', ylabel=r'$\log_{10} M_{200c}$',
              title='(2) inferred mass vs richness')
    ax[1].legend(fontsize=8)
    fig.suptitle(f'p(mass | richness) from {name} fitted params  '
                 f'[pi0={p["pi0"]:.3f}, Fm={p["Fm"]:.3f}, Gz={p["Gz"]:.3f}, '
                 f'sigl={p["sigl"]:.3f}]', fontsize=10)
    fig.tight_layout()
    out = join(EXPDIR, name, 'p_mass_given_lambda.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
    print(f'  prior width sig_pi={sig_pi:.3f}, posterior width s_post={s_post:.3f}'
          f'  (shrinkage from prior: {s_post/sig_pi:.2f}x)')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()
    full_corner(a.name)
    p_mass_given_lambda(a.name, seed=a.seed)
