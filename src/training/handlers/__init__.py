"""Training handlers - modular structure."""

from src.training.handlers.progress import attach_progress_bar_to_engine
from src.training.handlers.evaluation import attach_evaluator_handler
from src.training.handlers.wandb_logger import (
    attach_wandb_logger_to_trainer,
    attach_wandb_logger_to_evaluator
)
from src.training.handlers.checkpoint import (
    attach_checkpoint_handler_to_evaluator,
    attach_last_checkpoint_handler
)
from src.training.handlers.early_stopping import attach_early_stopping
from src.training.handlers.scheduler import attach_scheduler_handler

__all__ = [
    'attach_progress_bar_to_engine',
    'attach_evaluator_handler',
    'attach_wandb_logger_to_trainer',
    'attach_wandb_logger_to_evaluator',
    'attach_checkpoint_handler_to_evaluator',
    'attach_last_checkpoint_handler',
    'attach_early_stopping',
    'attach_scheduler_handler',
]

