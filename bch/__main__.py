"""Command line interface for bch SC2BA experiments."""

import argparse
from pathlib import Path

import torch

from .upper_state.influence import KINDS

DEFAULT_PLAN = Path(__file__).parent / "configs" / "probes.json"


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Record probes from the scenario and opponent schedule.")
    collect.add_argument("--horizon", type=positive, default=4)
    collect.add_argument("--behavior", action="append", help="Collect this probe behavior only. Repeat to select several.")
    fit = commands.add_parser("fit", help="Fit a state encoder and frozen medoids.")
    fit.add_argument("--probes", required=True, type=Path)
    fit.add_argument("--method", required=True, choices=KINDS[1:])
    fit.add_argument("--scenario", help="Fit from just this scenario. The default pools selected probe scenarios.")
    fit.add_argument("--behavior", action="append", help="Fit from this probe behavior only. Repeat to select several.")
    fit.add_argument("--epochs", type=positive, default=160)
    fit.add_argument("--codes", type=positive, default=32)
    fit.add_argument("--latent-dim", type=positive, default=32)
    fit.add_argument("--batch-size", type=positive, default=64)
    fit.add_argument("--codebook-samples", type=positive, default=4096)
    fit.add_argument("--device", default="cpu")
    train = commands.add_parser("train", help="Train both Q layers on the training layouts.")
    train.add_argument("--method", choices=KINDS, default="im-hash")
    train.add_argument("--encoder", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen policy on held-out layouts.")
    evaluate.add_argument("--policy", required=True, type=Path)
    evaluate.add_argument("--encoder", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    for command in (collect, fit, train, evaluate):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--seed", required=True, type=int)
        command.add_argument("--threads", type=positive, default=1, help="CPU threads for Torch and clustering.")
    for command in (collect, train, evaluate):
        command.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
        command.add_argument("--scenario", action="append", required=command is train,
                             help="Run only this scenario. Repeat for multiple scenarios.")
        repeats_unit = "layout/opponent/behavior" if command is collect else "layout/opponent"
        command.add_argument("--repeats", type=positive, default=1, help=f"Episodes per {repeats_unit} case.")
        command.add_argument("--step-mul", type=positive, default=8)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be a nonnegative 32-bit integer.")
    if args.command == "fit":
        if args.codebook_samples < args.codes:
            parser.error("--codebook-samples must be at least --codes.")
    torch.set_num_threads(args.threads)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=args.threads):
        if args.command == "fit":
            from .upper_state.fit import fit as fit_models

            fit_models(args)
        else:
            from .plan import read_plan

            document, available, groupings, behaviors = read_plan(args.plan)
            names = args.scenario or list(available)
            if len(names) != len(set(names)) or not set(names).issubset(available):
                parser.error(f"Choose distinct scenarios from {', '.join(available)}.")
            if args.command == "train" and len(names) != 1:
                parser.error("Train one scenario per Q checkpoint.")
            scenarios = [available[name] for name in names]
            if args.command == "collect":
                from .collect import collect as collect_probes

                chosen = args.behavior or list(behaviors)
                if len(chosen) != len(set(chosen)) or not set(chosen).issubset(behaviors):
                    parser.error(f"Choose distinct probe behaviors from {', '.join(behaviors)}.")
                collect_probes(args, document, scenarios, groupings, {name: behaviors[name] for name in chosen})
            else:
                from .run_policy import run_policy

                run_policy(args, document, scenarios)


if __name__ == "__main__":
    main()
