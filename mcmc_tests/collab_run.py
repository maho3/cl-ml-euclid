"""Run all COLLAB-report fits + figures. gnn_npe @ dC100 throughout.

M1 inverse | M2 forward(flat) | M3 forward p(m|z) | M4 calibration (rho=0) |
M5 joint bivariate (correlated residuals). Calibration models use cal_frac=0.1.
All bivariate fits are SINGLE-STEP (joint over relation + covariance) for
consistency; the two-step cut is unstable at cal_frac=0.1 (see report).

Saves samples to collab_results/<tag>.npz and figures to plots/.
"""
import os
import json

import numpy as np

from .data import Data, M0_PIV
from .legacy_models import fit_legacy
from .fit import run_candidate, run_reference
from .likelihoods.forward_cal import ForwardCal
from .two_stage import stage1_calibrate, run_stage2
from .compare import tension
from .collab_plots import corner_true_pred, scaling_plot, PLOTS

RES = os.path.join(os.path.dirname(__file__), 'collab_results')
os.makedirs(RES, exist_ok=True)
SET = dict(num_warmup=500, num_samples=1500, num_chains=4)
# lighter settings for the final-relations showcase (degenerate channels are slow
# and their contours are broad; precision not needed)
FINAL_SET = dict(num_warmup=400, num_samples=800, num_chains=2)
CAL_FRAC = 0.1
SEED = 3


def _save(tag, cand, ref, extra=None):
    d = {f'cand_{k}': np.asarray(v) for k, v in cand.items()}
    d.update({f'ref_{k}': np.asarray(v) for k, v in ref.items()})
    np.savez(os.path.join(RES, f'{tag}.npz'), **d)
    if extra:
        json.dump(extra, open(os.path.join(RES, f'{tag}.json'), 'w'), indent=2,
                  default=float)


def _dtxt(cand, ref):
    try:
        return f"d={tension(cand, ref)['d']:.2f}"
    except Exception:
        return ''


def run_legacy(which, tag, keys, kind, title):
    D = Data('gnn_npe', 'dC100')               # full sample (no cal needed)
    leg = dict(warmup=SET['num_warmup'], samples=SET['num_samples'],
               chains=SET['num_chains'])
    cand = fit_legacy(which, D, masses='pred', **leg)
    ref = fit_legacy(which, D, masses='true', **leg)
    _save(tag, cand, ref)
    corner_true_pred(cand, ref, keys, os.path.join(PLOTS, f'{tag}_corner.png'),
                     title=title)
    scaling_plot(D, cand, ref, os.path.join(PLOTS, f'{tag}_scaling.png'),
                 kind=kind, title=title, step=3)
    print(f'[{tag}] done '
          + ' '.join(f'{k}={np.median(cand[k]):.2f}/{np.median(ref[k]):.2f}'
                     for k in keys))


def run_calibration(tag, title):
    """M4: forward calibration, linear map, rho=0 (Gaussian channel)."""
    D = Data('gnn_npe', 'dC100', cal_frac=CAL_FRAC, seed=SEED)
    lik = ForwardCal(map='linear', channel='gaussian')
    cand, _ = run_candidate(D, lik, M0_PIV, M0_PIV, settings=SET, seed=SEED)
    ref, _ = run_reference(D, M0_PIV, settings=SET, seed=SEED + 1)
    _save(tag, cand, ref, extra=dict(d=tension(cand, ref)['d']))
    keys = ['pi0', 'Fm', 'Gz']
    corner_true_pred(cand, ref, keys, os.path.join(PLOTS, f'{tag}_corner.png'),
                     title=f'{title}  ({_dtxt(cand, ref)})')
    scaling_plot(D, cand, ref, os.path.join(PLOTS, f'{tag}_scaling.png'),
                 kind='forward', title=title, step=3)
    print(f'[{tag}] {_dtxt(cand, ref)} '
          f'Fm={np.median(cand["Fm"]):.3f}/{np.median(ref["Fm"]):.3f}')
    return D, ref


def run_bivariate_joint(tag, title, D=None, ref=None, model='gnn_npe',
                        settings=None, max_tree_depth=7):
    """M5 / final relations: bivariate-covariance model fit JOINTLY (relation +
    covariance in one MCMC), with the residual correlation rho FIXED at its
    calibration-subset estimate. Fixing rho removes the F_m-rho degeneracy, which
    the main sample can't identify and which otherwise breaks weak channels
    (msig/mamp); well-determined channels are unaffected (their posterior rho
    equals the cal value). cal richness enters only via the covariance
    (nuisance-absorbed mean, constraint #2)."""
    from .likelihoods.bivariate import BivariateForwardCal
    settings = settings or FINAL_SET
    if D is None:
        D = Data(model, 'dC100', cal_frac=CAL_FRAC, seed=SEED)
    fixed = stage1_calibrate(D)               # rho measured on the cal subset
    # heteroscedastic estimator scatter omega_i^2 = omega0^2 + kappa^2 sigma_i^2
    # uses the per-cluster posterior width; rho stays on the mean residual.
    lik = BivariateForwardCal(map='linear', width='hetero',
                              rho_prior=float(fixed['rho']))
    cand, _ = run_candidate(D, lik, M0_PIV, M0_PIV, settings=settings, seed=SEED,
                            max_tree_depth=max_tree_depth)
    if ref is None:
        ref, _ = run_reference(D, M0_PIV, settings=settings, seed=SEED + 1)
    _save(tag, cand, ref, extra=dict(d=tension(cand, ref)['d'],
                                     rho_cal=fixed['rho'],
                                     se_rho_cal=fixed['se_rho'],
                                     N_cal=fixed['N_cal']))
    keys = ['pi0', 'Fm', 'Gz']
    corner_true_pred(cand, ref, keys, os.path.join(PLOTS, f'{tag}_corner.png'),
                     title=f'{title}  ({_dtxt(cand, ref)})')
    scaling_plot(D, cand, ref, os.path.join(PLOTS, f'{tag}_scaling.png'),
                 kind='forward', title=title, step=3)
    print(f'[{tag}] {_dtxt(cand, ref)} '
          f'Fm={np.median(cand["Fm"]):.3f}/{np.median(ref["Fm"]):.3f} '
          f'rho={np.median(cand["rho"]):.3f} (cal {fixed["rho"]:.3f})')


def main_narrative():
    run_legacy('inverse', 'M1_inverse', ['A', 'B', 'C'], 'inverse',
               'M1: inverse model')
    run_legacy('forward', 'M2_forward', ['pi0', 'Fm', 'Gz'], 'forward',
               'M2: forward model, flat p(m)')
    run_legacy('forward_zprior', 'M3_zprior', ['pi0', 'Fm', 'Gz'], 'forward',
               'M3: forward model, p(m|z)')
    D, ref = run_calibration('M4_calib', 'M4: calibrated, no covariance')
    run_bivariate_joint('M5_bivariate', 'M5: calibrated + correlated residuals',
                        D=D, ref=ref, settings=SET, max_tree_depth=None)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'narrative':
        main_narrative()
