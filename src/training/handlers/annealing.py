"""Annealing handler for losses with epoch-dependent coefficients."""

from ignite.engine import Events


def attach_annealing_handler(trainer, criterion):
    """Tick criterion.current_epoch after each epoch.

    Args:
        trainer: Training engine
        criterion: Loss object with a current_epoch attribute
    """
    @trainer.on(Events.EPOCH_COMPLETED)
    def step_annealing(engine):
        criterion.current_epoch = engine.state.epoch
