"""Output bias shift matching a pretrained Dirichlet head to a prior mode."""

import math

import torch
import torch.nn.functional as F


def _final_linear(model):
    """Return the model's last nn.Linear, whose pre-activations feed softplus."""
    final = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            final = module

    if final is None:
        raise ValueError("No nn.Linear layer found in model.")
    if final.bias is None:
        raise ValueError("Final Linear layer has no bias to shift.")
    return final


def _probe_pre_activations(model, layer, loader, device, num_batches):
    """Return the layer's pre-activations, checking the model turns them into alpha."""
    cached, alphas = [], []
    hook = layer.register_forward_hook(lambda module, args, output: cached.append(output.detach()))

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            alphas.append(model(x.to(device)))
    hook.remove()

    z, alpha = torch.cat(cached), torch.cat(alphas)
    if not torch.isfinite(z).all():
        raise ValueError("Pre-activations are not finite; the pretrained checkpoint diverged.")
    if z.shape != alpha.shape or not torch.allclose(alpha, F.softplus(z) + 1.0):
        raise ValueError(
            "Model output is not softplus(last nn.Linear) + 1, so shifting that bias would "
            "not control alpha_0: the last Linear is not the Dirichlet head, or the head "
            "applies a different activation."
        )
    return z


def shift_output_bias_to_prior_mode(model, loader, prior_mode, device,
                                    num_batches=5, tol=1e-4, max_iter=60):
    """
    Add one constant to the output bias so median alpha_0 equals prior_mode.

    With alpha = softplus(z) + 1, the same constant b on every output bias
    raises alpha_0 = sum_c alpha_c monotonically and leaves the argmax
    unchanged, so it re-levels a pretrained head without changing what it
    predicts. b is bisected on the probe pre-activations; substituting their
    largest (smallest) value for all of them brackets it, since that makes
    alpha_0 an upper (lower) bound for every input.

    Args:
        model: Model with a Dirichlet output layer
        loader: DataLoader to probe (the training loader)
        prior_mode: Target median alpha_0; must exceed the number of classes,
            since alpha_c >= 1 puts alpha_0 >= num_classes for every input
        device: Device to run on
        num_batches: Number of probe batches
        tol: Relative tolerance on the achieved median alpha_0
        max_iter: Maximum number of bisection steps

    Returns:
        dict: 'bias_shift_b' (the applied constant), 'clip_fraction' (share of
            (input, class) pairs whose prior mean mass prior_mode * m_c falls
            below the alpha_c >= 1 floor), and 'prior_mode_target'

    Raises:
        ValueError: If prior_mode <= num_classes, if the probed pre-activations
            are not finite, or if the model's output is not softplus(z) + 1
    """
    layer = _final_linear(model)
    num_classes = layer.out_features
    if prior_mode <= num_classes:
        raise ValueError(
            f"prior_mode={prior_mode} is unreachable: alpha_c = softplus(z_c) + 1 >= 1 "
            f"puts alpha_0 >= num_classes = {num_classes} for every input."
        )

    z = _probe_pre_activations(model, layer, loader, device, num_batches)

    def median_alpha0(shift):
        return (F.softplus(z + shift) + 1.0).sum(dim=-1).median().item()

    # Bracket: at_target is the pre-activation at which every class contributes
    # an equal share of prior_mode, so measuring it against the largest
    # (smallest) probe pre-activation brackets b from below (above).
    per_class = (prior_mode - num_classes) / num_classes
    at_target = per_class + math.log1p(-math.exp(-per_class))  # inverse softplus
    lo, hi = at_target - z.max().item(), at_target - z.min().item()

    b = 0.5 * (lo + hi)
    for _ in range(max_iter):
        achieved = median_alpha0(b)
        if abs(achieved - prior_mode) <= tol * prior_mode:
            break
        if achieved < prior_mode:
            lo = b
        else:
            hi = b
        b = 0.5 * (lo + hi)

    with torch.no_grad():
        layer.bias.add_(b)

    alpha = F.softplus(z + b) + 1.0
    alpha0 = alpha.sum(dim=-1, keepdim=True)
    clip_fraction = (prior_mode * alpha / alpha0 < 1.0).float().mean().item()
    print(f"Output bias shift: b={b:.4f}, median alpha_0 {alpha0.median().item():.2f} "
          f"(target {prior_mode:.2f}), clip_fraction={clip_fraction:.4f}")

    return {'bias_shift_b': b, 'clip_fraction': clip_fraction,
            'prior_mode_target': prior_mode}
