"""Run experiments defined in experiments.yaml in declaration order.

State is persisted in .experiment_state.json so interrupted runs resume cleanly.

Usage:
    python run_experiments.py
    python run_experiments.py --only cifar10_bnn_T_0.01 cifar10_bnn_T_0.1
    python run_experiments.py --skip cifar10_bnn_T_1_no_aug
    python run_experiments.py --rerun cifar10_dirichlet_bnn
    python run_experiments.py --list
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

MANIFEST = Path("experiments.yaml")
STATE_FILE = Path(".experiment_state.json")
OUTPUT_ROOT = Path("outputs")


def load_manifest():
    raw = yaml.safe_load(MANIFEST.read_text())
    return raw.get("wandb_project"), raw["experiments"]


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_experiment(name, exp, state, wandb_project):
    output_dir = OUTPUT_ROOT / name

    overrides = [f"hydra.run.dir={output_dir}"]

    if dep := exp.get("pretrained_from"):
        if dep not in state:
            raise RuntimeError(f"'{name}' depends on '{dep}' which has not completed yet.")
        path = Path(state[dep]) / "best_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"No best_model.pt in {state[dep]}")
        overrides += ["training.pretrained.enabled=true", f"training.pretrained.path={path}"]

    if wandb_project:
        overrides.append(f"training.wandb.project={wandb_project}")
    overrides += exp.get("overrides", [])

    cmd = ["python", "train.py", f"--config-name={exp['config']}"] + overrides
    print(f"\nRunning {name}: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: '{name}' failed with exit code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)

    state[name] = str(output_dir)
    save_state(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument("--skip", nargs="+", metavar="NAME")
    parser.add_argument("--rerun", nargs="+", metavar="NAME")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    wandb_project, manifest = load_manifest()
    state = load_state()

    if args.list:
        for name, exp in manifest.items():
            dep = f"  [needs: {exp['pretrained_from']}]" if exp.get("pretrained_from") else ""
            print(f"  [{'done' if name in state else 'pending':7s}] {name}{dep}")
        return

    if args.rerun:
        for name in args.rerun:
            state.pop(name, None)
        save_state(state)

    skip = set(args.skip or [])
    to_run = args.only if args.only else list(manifest.keys())

    for name in to_run:
        if name not in manifest:
            print(f"Unknown experiment: {name}", file=sys.stderr)
            sys.exit(1)
        if name in skip or name in state:
            print(f"Skipping {name}" + (f" ({state[name]})" if name in state else ""))
            continue
        run_experiment(name, manifest[name], state, wandb_project)


if __name__ == "__main__":
    main()
