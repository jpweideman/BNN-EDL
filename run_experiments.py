"""Run experiments defined in experiments.yaml in declaration order.

State is persisted in .experiment_state.json so interrupted runs resume cleanly.

Usage:
    python run_experiments.py
    python run_experiments.py --runs 5
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
    return raw.get("wandb_project"), raw.get("seed"), raw["experiments"]


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_experiment(name, exp, state, wandb_project, seed):
    key = f"{name}_s{seed}"
    output_dir = OUTPUT_ROOT / key

    overrides = [f"hydra.run.dir={output_dir}"]

    if dep := exp.get("pretrained_from"):
        dep_key = f"{dep}_s{seed}"
        if dep_key not in state:
            raise RuntimeError(f"'{name}' depends on '{dep}' which has not completed yet.")
        path = Path(state[dep_key]) / "best_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"No best_model.pt in {state[dep_key]}")
        overrides += ["training.pretrained.enabled=true", f"training.pretrained.path={path}"]

    overrides.append(f"seed={seed}")
    if wandb_project:
        overrides.append(f"training.wandb.project={wandb_project}")

    exp_overrides = list(exp.get("overrides", []))
    idx = next((i for i, o in enumerate(exp_overrides) if o.startswith("training.wandb.name=")), None)
    if idx is not None:
        exp_overrides[idx] += f"_s{seed}"
    else:
        exp_overrides.append(f"training.wandb.name={name}_s{seed}")
    overrides += exp_overrides

    cmd = ["python", "train.py", f"--config-name={exp['config']}"] + overrides
    print(f"\nRunning {key}: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: '{key}' failed with exit code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)

    state[key] = str(output_dir)
    save_state(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, metavar="N")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument("--skip", nargs="+", metavar="NAME")
    parser.add_argument("--rerun", nargs="+", metavar="NAME")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    wandb_project, base_seed, manifest = load_manifest()
    state = load_state()
    base_seed = base_seed or 0

    if args.list:
        for name, exp in manifest.items():
            dep = f"  [needs: {exp['pretrained_from']}]" if exp.get("pretrained_from") else ""
            done = [k for k in state if k.startswith(f"{name}_s")]
            status = f"{len(done)} done" if done else "pending"
            print(f"  [{status:10s}] {name}{dep}")
        return

    if args.rerun:
        for name in args.rerun:
            for k in list(state.keys()):
                if k == name or k.startswith(f"{name}_s"):
                    state.pop(k)
        save_state(state)

    skip = set(args.skip or [])
    to_run = args.only if args.only else list(manifest.keys())

    for run_idx in range(args.runs):
        seed = base_seed + run_idx
        for name in to_run:
            if name not in manifest:
                print(f"Unknown experiment: {name}", file=sys.stderr)
                sys.exit(1)
            key = f"{name}_s{seed}"
            if name in skip or key in state:
                print(f"Skipping {key}" + (f" ({state[key]})" if key in state else ""))
                continue
            run_experiment(name, manifest[name], state, wandb_project, seed)


if __name__ == "__main__":
    main()
