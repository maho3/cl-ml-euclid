"""Final-relations section: joint (single-step) bivariate model for the six mass
models (msig, pamico/M-lambdaspec, mamp, gals_nle, summ_nle, gnn_npe) at
cal_frac=0.1, plus a per-model rho diagnostic (corr of mass/richness residuals at
fixed true mass) -- supports 'models that read richness have high rho'. Same
fitting (single-step joint) as M5 for consistency.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data import Data, M0_PIV
from .collab_run import run_bivariate_joint
from .collab_plots import PLOTS

MODELS = [('msig', r'$M$--$\sigma$'), ('pamico', r'$M$--$\lambda_{\rm spec}$'),
          ('mamp', 'MAMPOSSt'), ('gals_nle', 'Galaxy-Net'),
          ('summ_nle', 'Summary-Net'), ('gnn_npe', 'Graph-Net')]


def rho_diagnostic():
    """corr(mu-resid, ell-resid | m_true, z) for each model -> bar chart."""
    rhos = {}
    for m, _ in MODELS:
        D = Data(m, 'dC100')
        mu, ell, mt, zeta = D.mu, D.ell, D.mtrue, D.zeta
        dm = mt - M0_PIV
        Amu = np.vstack([np.ones_like(dm), dm, dm ** 2]).T
        rmu = mu - Amu @ np.linalg.lstsq(Amu, mu, rcond=None)[0]
        Ael = np.vstack([np.ones_like(dm), dm, zeta]).T
        rel = ell - Ael @ np.linalg.lstsq(Ael, ell, rcond=None)[0]
        rhos[m] = float(np.corrcoef(rmu, rel)[0, 1])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    names = [t for _, t in MODELS]
    vals = [rhos[m] for m, _ in MODELS]
    # red = NLE models that ingest galaxy content; blue = dynamical/other
    reads = {'gals_nle', 'summ_nle', 'gnn_npe'}
    cols = ['C3' if m in reads else 'C0' for m, _ in MODELS]
    ax.bar(names, vals, color=cols)
    ax.tick_params(axis='x', labelsize=8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel(r'corr$(\hat m,\ \lambda_{\rm phot}\ |\ m_{\rm true}, z)$')
    ax.set_title('Residual correlation by mass model')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'final_rho_by_model.png'), dpi=130)
    plt.close(fig)
    print('rho by model:', {m: round(rhos[m], 3) for m, _ in MODELS})
    return rhos


def main():
    rho_diagnostic()
    for m, title in MODELS:
        run_bivariate_joint(f'final_{m}', f'{title} (joint bivariate)', model=m)


if __name__ == '__main__':
    main()
