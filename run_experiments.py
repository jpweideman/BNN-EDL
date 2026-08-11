"""Run experiments defined in an experiments yaml in declaration order.

State is persisted per experiments file (experiments_e1.yaml ->
.experiments_e1_state.json), so interrupted runs resume cleanly and each
experiment list tracks its own progress.

Entries warm-start from another entry's best checkpoint with pretrained_from,
of the same seed or, with pretrained_from_next_seed, of the next seed cyclic
over --runs. Cross-seed entries take a second pass over the seeds, once every
seed's source run exists, so one invocation runs the whole list.

Usage (--only, --skip and --rerun all take one or more experiment names):
    python run_experiments.py --file_name experiments_e1.yaml --runs 3
    python run_experiments.py --file_name experiments_e1.yaml --only e1_fm_L100_shift
    python run_experiments.py --file_name experiments_e1.yaml --list
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

OUTPUT_ROOT = Path("outputs")


def load_experiments(experiments_file):
    raw = yaml.safe_load(experiments_file.read_text())
    return raw.get("wandb_project"), raw.get("seed"), raw["experiments"]


def load_state(state_file):
    return json.loads(state_file.read_text()) if state_file.exists() else {}


def save_state(state_file, state):
    state_file.write_text(json.dumps(state, indent=2))


def run_experiment(name, exp, state, state_file, wandb_project, seed, dep_seed):
    key = f"{name}_s{seed}"
    output_dir = OUTPUT_ROOT / key

    overrides = [f"hydra.run.dir={output_dir}"]

    if dep := exp.get("pretrained_from"):
        dep_key = f"{dep}_s{dep_seed}"
        if dep_key not in state:
            raise RuntimeError(f"'{key}' depends on '{dep_key}' which has not completed yet.")
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
    save_state(state_file, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", required=True, metavar="YAML",
                        help="experiments yaml to run")
    parser.add_argument("--runs", type=int, default=1, metavar="N")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument("--skip", nargs="+", metavar="NAME")
    parser.add_argument("--rerun", nargs="+", metavar="NAME")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    experiments_file = Path(args.file_name)
    if not experiments_file.exists():
        print(f"No such file: {experiments_file}", file=sys.stderr)
        sys.exit(1)
    state_file = Path(f".{experiments_file.stem}_state.json")

    wandb_project, base_seed, experiments = load_experiments(experiments_file)
    state = load_state(state_file)
    base_seed = base_seed or 0

    if args.list:
        for name, exp in experiments.items():
            seed_note = " of the next seed" if exp.get("pretrained_from_next_seed") else ""
            dep = f"  [needs: {exp['pretrained_from']}{seed_note}]" if exp.get("pretrained_from") else ""
            done = [k for k in state if k.startswith(f"{name}_s")]
            status = f"{len(done)} done" if done else "pending"
            print(f"  [{status:10s}] {name}{dep}")
        return

    if args.rerun:
        for name in args.rerun:
            for k in list(state.keys()):
                if k == name or k.startswith(f"{name}_s"):
                    state.pop(k)
        save_state(state_file, state)

    skip = set(args.skip or [])
    to_run = args.only if args.only else list(experiments.keys())
    for name in to_run:
        if name not in experiments:
            print(f"Unknown experiment: {name}", file=sys.stderr)
            sys.exit(1)

    # Entries warm-started from the next seed can only run once every seed's
    # source run exists, so they take a second pass over the seeds.
    cross_seed = [n for n in to_run if experiments[n].get("pretrained_from_next_seed")]
    for entries in ([n for n in to_run if n not in cross_seed], cross_seed):
        for run_idx in range(args.runs):
            seed = base_seed + run_idx
            for name in entries:
                key = f"{name}_s{seed}"
                if name in skip or key in state:
                    print(f"Skipping {key}" + (f" ({state[key]})" if key in state else ""))
                    continue
                dep_seed = base_seed + ((run_idx + 1) % args.runs) if name in cross_seed else seed
                run_experiment(name, experiments[name], state, state_file,
                               wandb_project, seed, dep_seed)


if __name__ == "__main__":
    main()
