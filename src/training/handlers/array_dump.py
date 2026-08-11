"""Array dump handler."""

import json
from pathlib import Path

import numpy as np
from ignite.engine import Events


def attach_array_dump_handler(trainer, evaluators, output_dir):
    """
    Write the per-input arrays and final scalar metrics of every evaluator.

    Runs at trainer completion, so it sees the final evaluations (including
    the interval=-1 splits). Writes <output_dir>/arrays/<split>.npz for every
    split with an array_dump metric, plus arrays/summary.json, which lets the
    analysis layer work offline from the run directory.

    Args:
        trainer: Training engine
        evaluators: Dict of evaluators from create_evaluators
        output_dir: Output directory for the arrays
    """
    @trainer.on(Events.COMPLETED)
    def dump_arrays(engine):
        arrays_dir = Path(output_dir) / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        summary = {}
        num_dumped = 0
        for split_name, eval_data in evaluators.items():
            metrics = eval_data['evaluator'].state.metrics
            if metrics:
                summary[split_name] = {k: float(v) for k, v in metrics.items()
                                       if isinstance(v, (int, float))}

            dump_metric = eval_data['metrics'].get('array_dump')
            if dump_metric is not None and dump_metric.has_data:
                np.savez_compressed(arrays_dir / f"{split_name}.npz", **dump_metric.data)
                num_dumped += 1

        (arrays_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Array dump: wrote summary.json and {num_dumped} npz file(s) to {arrays_dir}")
