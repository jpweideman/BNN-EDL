"""Objective utilities for metric optimization."""

def get_comparison_fn(objective):
    """
    Returns (initial_value, is_better_fn) for the given objective.
    
    Args:
        objective: Must be "maximize" or "minimize"
        
    Returns:
        tuple: (initial_best_value, comparison_function)
    """
    if objective == "minimize":
        return (float('inf'), lambda curr, best: curr < best)
    elif objective == "maximize":
        return (-float('inf'), lambda curr, best: curr > best)
    else:
        raise ValueError(f"objective must be 'maximize' or 'minimize', got '{objective}'")

def get_score_sign(objective):
    """
    Returns score multiplier for the given objective.
    
    Args:
        objective: Must be "maximize" or "minimize"
        
    Returns:
        int: -1 for minimize, 1 for maximize
    """
    if objective == "minimize":
        return -1
    elif objective == "maximize":
        return 1
    else:
        raise ValueError(f"objective must be 'maximize' or 'minimize', got '{objective}'")

def to_wandb_summary(objective):
    """
    Converts objective to W&B summary type.
    
    Args:
        objective: Must be "maximize" or "minimize"
        
    Returns:
        str: "min" or "max"
    """
    if objective == "minimize":
        return "min"
    elif objective == "maximize":
        return "max"
    else:
        raise ValueError(f"objective must be 'maximize' or 'minimize', got '{objective}'")

