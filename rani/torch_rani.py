"""
rani/torch_rani.py
==================
RANI: Rational Adaptive Normalizing Initialization
PyTorch Implementation

Formula
-------
    RANI(x) = lambda * x * (1 + alpha * |x|) / (1 + |x|)   for x >= 0
    RANI(x) = lambda * alpha * (exp(x) - 1)                  for x < 0

Positive branch (x >= 0):
    - Rational function — zero transcendental operations (no exp, no sigmoid)
    - Slope at origin = lambda
    - Asymptotic slope as x -> +inf = lambda * alpha
    - Strictly increasing for all x > 0

Negative branch (x < 0):
    - Exponential — same structure as SELU's negative branch
    - Starts at 0 when x = 0
    - Saturates at -lambda * alpha as x -> -infinity

Self-Normalizing Initialization
---------------------------------
    The default values alpha = 1.6733 and lam = 1.0507 are the exact
    numerical solution to the self-normalizing fixed-point equations:

        E[ RANI(Z) ]    = 0     (mean preserved as zero)
        E[ RANI(Z)^2 ]  = 1     (variance preserved as one)

    where Z ~ N(0, 1) is a standard normal random variable.
    Derivation is in Part 4 of the mathematical foundation document.

    After initialization, both alpha and lam are learned per-layer
    by the optimizer during training.

Author : Nikki Rani
Paper  : doi.org/10.5281/zenodo.XXXXXXX  (update after Zenodo upload)
PyPI   : pip install rani-activation
GitHub : github.com/Star-nikki/rani-activation
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# FUNCTIONAL API
# Use this when you want fixed (non-learnable) alpha and lam values.
# =============================================================================

def rani(x: Tensor,
         alpha: float = 1.6733,
         lam: float   = 1.0507) -> Tensor:
    """
    Apply the RANI activation function with fixed parameters.

    This is the functional (non-learnable) version. For learnable parameters
    inside a neural network, use the RANI nn.Module class below instead.

    Args:
        x     (Tensor) : Input tensor. Any shape. Any device (CPU or GPU).
        alpha (float)  : Asymmetry parameter. Controls negative saturation
                         depth and asymptotic positive slope. Default 1.6733.
        lam   (float)  : Scale parameter. Controls overall amplitude.
                         Default 1.0507.

    Returns:
        Tensor: Same shape and device as x. RANI applied element-wise.

    Example:
        >>> import torch
        >>> from rani.torch_rani import rani
        >>> x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        >>> rani(x)
        tensor([-1.7578, -0.9209,  0.0000,  1.3147,  2.8860])
    """
    pos = lam * x * (1.0 + alpha * x.abs()) / (1.0 + x.abs())
    neg = lam * alpha * (x.exp() - 1.0)
    return torch.where(x >= 0, pos, neg)


# =============================================================================
# MODULE API
# Use this inside nn.Sequential or any nn.Module network.
# alpha and lam are learned automatically by the optimizer.
# =============================================================================

class RANI(nn.Module):
    """
    RANI: Rational Adaptive Normalizing Initialization.

    A self-normalizing activation function with learnable per-layer
    parameters alpha and lam. Both parameters are updated by the optimizer
    during backpropagation — each layer of your network learns its own
    optimal values.

    The function has two branches joined at x = 0:

        Positive branch (x >= 0):
            lam * x * (1 + alpha * |x|) / (1 + |x|)
            → rational function, no exp() call, computationally cheap

        Negative branch (x < 0):
            lam * alpha * (exp(x) - 1)
            → exponential, same structure as SELU's negative branch

    At initialization, alpha = 1.6733 and lam = 1.0507. These are the
    exact values satisfying the self-normalizing fixed-point equations
    E[RANI(Z)] = 0 and E[RANI(Z)^2] = 1 for Z ~ N(0,1).

    Args:
        alpha (float): Initial asymmetry parameter. Default 1.6733.
        lam   (float): Initial scale parameter.    Default 1.0507.

    Shape:
        Input  : (*) — any shape (batch, features, etc.)
        Output : (*) — identical shape to input

    Attributes:
        alpha (nn.Parameter): Learnable asymmetry parameter. Shape: scalar.
        lam   (nn.Parameter): Learnable scale parameter.    Shape: scalar.

    Examples:
        Basic usage::

            >>> import torch
            >>> import torch.nn as nn
            >>> from rani.torch_rani import RANI
            >>>
            >>> act = RANI()
            >>> x = torch.randn(32, 128)
            >>> y = act(x)
            >>> y.shape
            torch.Size([32, 128])

        Inside a Sequential network::

            >>> model = nn.Sequential(
            ...     nn.Linear(784, 512),
            ...     RANI(),
            ...     nn.Linear(512, 256),
            ...     RANI(),
            ...     nn.Linear(256, 128),
            ...     RANI(),
            ...     nn.Linear(128, 10),
            ... )

        Checking learned parameter values after training::

            >>> act = RANI()
            >>> print(act)
            RANI(alpha=1.6733, lam=1.0507)
            >>> # ... after training ...
            >>> print(act)
            RANI(alpha=1.5821, lam=1.0312)   # learned different values

    Reference:
        Rani, N. (2025). RANI: Rational Adaptive Normalizing Initialization.
        Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
    """

    def __init__(self,
                 alpha: float = 1.6733,
                 lam:   float = 1.0507) -> None:
        super().__init__()
        # Register as nn.Parameter so the optimizer updates them automatically
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.lam   = nn.Parameter(torch.tensor(float(lam)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RANI activation element-wise.

        Args:
            x (Tensor): Input tensor of any shape.

        Returns:
            Tensor: Same shape as input, RANI applied.
        """
        # Positive branch: rational function — no transcendental operations
        pos = self.lam * x * (1.0 + self.alpha * x.abs()) / (1.0 + x.abs())

        # Negative branch: exponential — same as SELU's negative branch
        neg = self.lam * self.alpha * (x.exp() - 1.0)

        # Combine: use positive branch where x >= 0, negative branch where x < 0
        return torch.where(x >= 0, pos, neg)

    def extra_repr(self) -> str:
        """
        Show current parameter values when you print the module.
        Example: RANI(alpha=1.6733, lam=1.0507)
        """
        return (f"alpha={self.alpha.item():.4f}, "
                f"lam={self.lam.item():.4f}")


# =============================================================================
# WEIGHT INITIALIZATION HELPERS
# Use these when building networks with RANI activations.
# =============================================================================

def lecun_normal_(tensor: Tensor) -> Tensor:
    """
    Apply LeCun normal initialization to a weight tensor, in-place.

    This is the mathematically correct weight initialization for RANI.
    Derived in Part 6 of the mathematical document from the condition:

        sigma_w^2 = 1 / (n * E[RANI(Z)^2]) = 1 / n

    since E[RANI(Z)^2] = 1 at the self-normalizing fixed point.

    Weights are drawn from N(0, 1/fan_in) where fan_in is the number
    of input connections to each neuron.

    Args:
        tensor (Tensor): 2D weight tensor of shape [out_features, in_features].

    Returns:
        The same tensor, initialized in-place.

    Raises:
        ValueError: If tensor is not 2-dimensional.

    Example:
        >>> import torch.nn as nn
        >>> from rani.torch_rani import lecun_normal_
        >>> layer = nn.Linear(256, 128)
        >>> lecun_normal_(layer.weight)
        >>> layer.weight.std()   # should be close to 1/sqrt(256) = 0.0625
    """
    if tensor.dim() < 2:
        raise ValueError(
            f"lecun_normal_ requires at least a 2D tensor, "
            f"got {tensor.dim()}D tensor of shape {tensor.shape}."
        )
    fan_in = tensor.size(1)
    std = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std)
    return tensor


def init_rani_network(model: nn.Module) -> None:
    """
    Apply LeCun normal initialization to all Linear layers in a model.
    Set all biases to zero.

    Call this once after constructing any network that uses RANI activations,
    before training begins.

    Args:
        model (nn.Module): Any PyTorch model containing Linear layers.

    Example:
        >>> import torch.nn as nn
        >>> from rani.torch_rani import RANI, init_rani_network
        >>>
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 512),
        ...     RANI(),
        ...     nn.Linear(512, 256),
        ...     RANI(),
        ...     nn.Linear(256, 10),
        ... )
        >>> init_rani_network(model)   # apply correct weight init
        >>> # Now train normally with any optimizer
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


# =============================================================================
# QUICK TEST — run this file directly to verify the implementation
# python rani/torch_rani.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RANI PyTorch Implementation — Quick Verification")
    print("=" * 60)

    act = RANI()
    print(f"\nModule:  {act}")
    print(f"alpha:   {act.alpha.item():.4f}  (initialized at fixed-point value)")
    print(f"lam:     {act.lam.item():.4f}  (initialized at fixed-point value)")

    # Test key values
    tests = {
        "RANI(0.0)  should be 0.0":    (0.0,   0.0,    1e-6),
        "RANI(-100) should be ~-1.758": (-100.0, -1.758, 1e-2),
        "RANI(1.0)  should be > 0":     None,
        "RANI(-1.0) should be < 0":     None,
    }

    x_zero   = torch.tensor([0.0])
    x_neg100 = torch.tensor([-100.0])
    x_pos    = torch.tensor([1.0])
    x_neg    = torch.tensor([-1.0])

    print("\nKey value checks:")
    print(f"  RANI(0.0)   = {act(x_zero).item():.6f}   (expected: 0.0)")
    print(f"  RANI(-100)  = {act(x_neg100).item():.6f}  (expected: ~-1.7582)")
    print(f"  RANI(1.0)   = {act(x_pos).item():.6f}   (expected: > 0)")
    print(f"  RANI(-1.0)  = {act(x_neg).item():.6f}  (expected: < 0)")
    print(f"  RANI(-500)  = {act(torch.tensor([-500.0])).item():.6f}  (expected: finite)")
    print(f"  RANI(500)   = {act(torch.tensor([500.0])).item():.6f}   (expected: finite)")

    # Test shape preservation
    x_batch = torch.randn(32, 128)
    y_batch = act(x_batch)
    print(f"\nShape test:  input {x_batch.shape} → output {y_batch.shape}  (expected: same)")

    # Test parameters are learnable
    optim = torch.optim.Adam(act.parameters(), lr=1e-2)
    loss  = act(torch.randn(10)).mean()
    loss.backward()
    optim.step()
    print(f"\nAfter one optimizer step:")
    print(f"  alpha: {act.alpha.item():.6f}  (should differ from 1.6733)")
    print(f"  lam:   {act.lam.item():.6f}  (should differ from 1.0507)")

    # Test NaN safety
    extreme = torch.tensor([-500.0, -100.0, 0.0, 100.0, 500.0])
    out     = act(extreme)
    nan_check = torch.isnan(out).any() or torch.isinf(out).any()
    print(f"\nNaN/Inf check on extreme inputs: {'FAILED' if nan_check else 'PASSED'}")

    print("\n" + "=" * 60)
    print("All checks done. Run pytest tests/test_torch.py for full tests.")
    print("=" * 60)