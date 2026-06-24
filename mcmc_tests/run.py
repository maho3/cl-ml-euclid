"""CLI / experiment driver for the forward calibration search.

Usage:
    python -m mcmc_tests.run --name v7_quad --map quadratic --summary mean
                              [--model gnn_npe --data dC100 --full]

Each run writes experiments/<name>/ with config.json, samples.npz, compare.json,
corner.png, checks.png, log.md and appends a row to RESULTS.md.
"""
import argparse
import json
import os
import subprocess
from os.path import join

import numpy as np

from .data import Data, M0_PIV
from .checks import diagnose, predict_bias
from .fit import run_candidate, run_reference, FAST, FULL
from .gates import grid_vs_closed, speed
from .compare import tension, save_compare, corner_overlay
from .likelihoods.forward_cal import ForwardCal

EXPDIR = join(os.path.dirname(__file__), 'experiments')
RESULTS = join(os.path.dirname(__file__), 'RESULTS.md')


def git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return 'unknown'


def append_results_row(row):
    header = ('| name | map | summary | b_fwd | quad_c2 | skew_med | '
              'predicted | d | PASS | note |\n'
              '|---|---|---|---|---|---|---|---|---|---|\n')
    if not os.path.exists(RESULTS):
        with open(RESULTS, 'w') as f:
            f.write('# Forward-calibration experiment ledger\n\n' + header)
    with open(RESULTS, 'a') as f:
        f.write(row + '\n')


def experiment(name, model='gnn_npe', data='dC100', summary='mean',
               map='quadratic', z_term=False, width='v7', channel='gaussian',
               rho_map='const', cal_frac=0.2, seed=0, full=False, grid_n=301,
               run_ref=True, ref_cache=None, note=''):
    settings = FULL if full else FAST
    outdir = join(EXPDIR, name)
    os.makedirs(outdir, exist_ok=True)
    m0 = m_ref = M0_PIV

    D = Data(model, data, summary=summary, cal_frac=cal_frac, seed=seed)
    if channel == 'bivariate':
        from .likelihoods.bivariate import BivariateForwardCal
        lik = BivariateForwardCal(map=map if map in ('linear', 'quadratic')
                                  else 'linear', rho_map=rho_map)
    else:
        lik = ForwardCal(map=map, z_term=z_term, width=width, channel=channel)

    # ---- (i) assumption checks first ----
    metrics = diagnose(D, m_ref=m_ref, fig_path=join(outdir, 'checks.png'))
    prediction = predict_bias(metrics, map=map)

    # ---- (ii) speed gate ----
    ms_eval, ms_grad = speed(D, lik, m0, m_ref)

    # ---- (iii) grid-vs-closed regression (linear sub-case) ----
    gvc = grid_vs_closed(D, m_ref, m0)

    # ---- (v) real fit ----
    cand, mc = run_candidate(D, lik, m0, m_ref, settings=settings, seed=seed)
    if run_ref:
        ref, mr = run_reference(D, m0, settings=settings, seed=seed + 1)
    else:
        ref = ref_cache

    res = tension(cand, ref)
    save_compare(res, join(outdir, 'compare.json'))
    corner_overlay(cand, ref, join(outdir, 'corner.png'),
                   title=f'{name}: d={res["d"]:.2f} '
                         f'({"PASS" if res["PASS"] else "FAIL"})')

    # save samples
    np.savez(join(outdir, 'samples.npz'),
             **{f'cand_{k}': np.asarray(v) for k, v in cand.items()},
             **{f'ref_{k}': np.asarray(v) for k, v in ref.items()})

    config = dict(name=name, model=model, data=data, summary=summary, map=map,
                  z_term=z_term, cal_frac=cal_frac, seed=seed, full=full,
                  grid_n=grid_n, settings=settings, git_sha=git_sha(),
                  N=D.N, N_cal=int(D.is_cal.sum()), N_main=int(D.sel.sum()),
                  ms_eval=ms_eval, ms_grad=ms_grad, grid_vs_closed=gvc,
                  metrics=metrics)
    with open(join(outdir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=float)

    # log.md
    pc = {k: np.percentile(cand[k], [16, 50, 84]) for k in ['pi0', 'Fm', 'Gz']}
    pr = {k: np.percentile(ref[k], [16, 50, 84]) for k in ['pi0', 'Fm', 'Gz']}
    with open(join(outdir, 'log.md'), 'w') as f:
        f.write(f'# {name}\n\n')
        f.write(f'model={model} data={data} summary={summary} map={map} '
                f'z_term={z_term} cal_frac={cal_frac} '
                f'settings={"FULL" if full else "FAST"}\n\n')
        f.write(f'## Assumption checks\n```\n{json.dumps(metrics, indent=2)}\n```\n')
        f.write(f'\n**Pre-fit prediction:** {prediction}\n\n')
        f.write(f'## Gates\nms/eval={ms_eval:.2f}, ms/grad={ms_grad:.2f}, '
                f'grid_vs_closed={gvc:.2e}\n\n')
        f.write('## Result\n')
        for k in ['pi0', 'Fm', 'Gz']:
            f.write(f'- {k}: cand {pc[k][1]:.3f} (+{pc[k][2]-pc[k][1]:.3f}/'
                    f'-{pc[k][1]-pc[k][0]:.3f})  vs  ref {pr[k][1]:.3f} '
                    f'(+{pr[k][2]-pr[k][1]:.3f}/-{pr[k][1]-pr[k][0]:.3f})\n')
        f.write(f'\n**d = {res["d"]:.3f} -> '
                f'{"PASS" if res["PASS"] else "FAIL"}**\n')
        if note:
            f.write(f'\nNote: {note}\n')

    append_results_row(
        f'| {name} | {map} | {summary} | {metrics["b_fwd"]:.2f} | '
        f'{metrics["quad_c2"]:.2f} | {metrics["skew_med"]:.2f} | '
        f'{prediction[:60]}... | {res["d"]:.3f} | '
        f'{"PASS" if res["PASS"] else "FAIL"} | {note} |')

    print(f'[{name}] d={res["d"]:.3f} {"PASS" if res["PASS"] else "FAIL"} '
          f'| ms/eval={ms_eval:.1f} gvc={gvc:.1e}')
    print(f'  prediction: {prediction}')
    return res, cand, ref, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', default='gnn_npe')
    ap.add_argument('--data', default='dC100')
    ap.add_argument('--summary', default='mean',
                    choices=['mean', 'median', 'mode'])
    ap.add_argument('--map', default='quadratic',
                    choices=['linear', 'quadratic', 'cubic'])
    ap.add_argument('--z-term', action='store_true')
    ap.add_argument('--channel', default='gaussian',
                    choices=['gaussian', 'skewnorm', 'bivariate'])
    ap.add_argument('--rho-map', default='const', choices=['const', 'linear'])
    ap.add_argument('--cal-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    experiment(a.name, model=a.model, data=a.data, summary=a.summary,
               map=a.map, z_term=a.z_term, channel=a.channel, rho_map=a.rho_map,
               cal_frac=a.cal_frac, seed=a.seed, full=a.full, note=a.note)


if __name__ == '__main__':
    main()
