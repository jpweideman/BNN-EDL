"""Evaluation handler."""

from ignite.engine import Events


def attach_evaluator_handler(trainer, evaluator, loader, name, interval):
    """
    Attach evaluation handler.
    
    Args:
        trainer: Training engine
        evaluator: Evaluation engine
        loader: Data loader
        name: Display name for this evaluation
        interval: Evaluation frequency (0=disabled, -1=last epoch only, >0=every N epochs)
    """
    if interval == 0:
        return
    
    if interval == -1:
        @trainer.on(Events.COMPLETED)
        def run_final_eval(engine):
            evaluator.run(loader)
            print(f"\nFinal {name} Evaluation:")
            for k, v in evaluator.state.metrics.items():
                print(f"  {k}: {v:.4f}")
    else:
        @trainer.on(Events.EPOCH_COMPLETED(every=interval))
        def run_periodic_eval(engine):
            evaluator.run(loader)
            print(f"\nEpoch {engine.state.epoch} - {name}:")
            for k, v in evaluator.state.metrics.items():
                print(f"  {k}: {v:.4f}")

