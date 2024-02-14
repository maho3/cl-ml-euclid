import argparse
import numpy as np
from ili.dataloaders import NumpyLoader
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('style.mcstyle')

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run SBI inference for toy data.")
    parser.add_argument('--model', type=str, default='summ')
    parser.add_argument('--data', type=str, default='dC100')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    train = f'./configs/dat/{args.model}_train.yaml'
    test = f'./configs/dat/{args.model}_test.yaml'
    inf = f'./configs/inf/{args.model}.yaml'
    val = f'./configs/val/{args.model}.yaml'

    # load training dataloader
    in_dir = f"./data/processed/AMICO{args.data}"
    if args.model == 'summ':
        x = np.load(f"{in_dir}/x_sum.npy", allow_pickle=True)
        theta = np.load(f"{in_dir}/theta_batch.npy", allow_pickle=True)
        folds = np.load(f"{in_dir}/folds_batch.npy", allow_pickle=True)
    elif args.model == 'gals':
        x = np.load(f"{in_dir}/x.npy", allow_pickle=True)
        theta = np.load(f"{in_dir}/theta.npy", allow_pickle=True)
        folds = np.load(f"{in_dir}/folds.npy", allow_pickle=True)
    mask = folds != args.fold
    train_loader = NumpyLoader(x[mask], theta[mask])

    # load test dataloader
    if args.model == 'gals':
        x = np.load(f"{in_dir}/x_batch.npy", allow_pickle=True)
        theta = np.load(f"{in_dir}/theta_batch.npy", allow_pickle=True)
        folds = np.load(f"{in_dir}/folds_batch.npy", allow_pickle=True)
    mask = folds == args.fold
    test_loader = NumpyLoader(x[mask], theta[mask])

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    out_dir = f"./saved_models/{args.model}_nle_{args.data}_f{args.fold}"
    runner = SBIRunner.from_config(
        inf, out_dir=out_dir)
    posterior, summary = runner(loader=train_loader)

    # plot the loss
    f = plt.figure()
    for i, x in enumerate(summary):
        plt.plot(x['training_log_probs'], label=f'train {i}', c=f'C{i}')
        plt.plot(x['validation_log_probs'],
                 label=f'val {i}', c=f'C{i}', ls='--')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Log probability')
    f.savefig(f"{out_dir}/loss.png")

    # # use the trained posterior model to predict on a single example from
    # # the test set
    val_runner = ValidationRunner.from_config(
        val, out_dir=out_dir)
    val_runner(loader=test_loader)
