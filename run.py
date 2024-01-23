import argparse
from ili.dataloaders import StaticNumpyLoader
from ili.inference.runner_sbi import SBIRunner
from ili.validation.runner import ValidationRunner
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run SBI inference for toy data.")
    parser.add_argument('--model', type=str, default='summ')
    parser.add_argument('--data', type=str, default='dC100')
    args = parser.parse_args()
    train = f'./configs/dat/{args.model}_train.yaml'
    test = f'./configs/dat/{args.model}_test.yaml'
    inf = f'./configs/inf/{args.model}.yaml'
    val = f'./configs/val/{args.model}.yaml'

    # reload all simulator examples as a dataloader
    train_loader = StaticNumpyLoader.from_config(
        train, in_dir=f"./data/processed/AMICO{args.data}")
    test_loader = StaticNumpyLoader.from_config(
        test, in_dir=f"./data/processed/AMICO{args.data}")

    # train a model to infer x -> theta. save it as toy/posterior.pkl
    runner = SBIRunner.from_config(
        inf, out_dir=f"./saved_models/{args.model}_nle_{args.data}")
    runner(loader=train_loader)

    # # use the trained posterior model to predict on a single example from
    # # the test set
    val_runner = ValidationRunner.from_config(
        val, out_dir=f"./saved_models/{args.model}_nle_{args.data}")
    val_runner(loader=test_loader)
