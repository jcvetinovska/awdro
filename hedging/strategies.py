import math
import torch
import numpy as np

class ConstantHedge:

    def __init__(self, holding):
        # Store the number of shares here
        self.holding = float(holding)

        if not math.isfinite(self.holding):
            raise ValueError("Strategy must be finite")


    def __call__(self, t, history):
        # Return one holding per path
        return torch.full(
            (history.shape[0],),
            self.holding,
            dtype=history.dtype,
            device=history.device,
        )


class AffineHedge:
    """Hold shares according to an affine function of stock and variance.

    Parameters
    ----------
    coefficients : array-like, shape (n_steps, 3)
        Row t contains [a_t, b_t, c_t], defining:
        H_t = a_t + b_t * (S_t / s0 - 1) + c_t * (V_t - v0).
    s0 : float
        Positive initial stock price.
    v0 : float
        Nonnegative initial variance.

    Notes
    -----
    Coefficients are fixed during evaluation. Holdings remain
    differentiable with respect to the supplied path observations.
    """

    def __init__(self, coefficients, s0, v0):
        self.coefficients = np.asarray(
            coefficients, dtype=float
        ).copy()
        self.s0 = float(s0)
        self.v0 = float(v0)

        if (
            self.coefficients.ndim != 2
            or self.coefficients.shape[0] == 0
            or self.coefficients.shape[1] != 3
        ):
            raise ValueError(
                "coefficients must have shape (n_steps, 3), "
                "with at least one trading interval."
            )

        if not np.isfinite(self.coefficients).all():
            raise ValueError("Coefficients must be finite.")

        if not math.isfinite(self.s0) or self.s0 <= 0:
            raise ValueError("s0 must be finite and positive.")

        if not math.isfinite(self.v0) or self.v0 < 0:
            raise ValueError("v0 must be finite and nonnegative.")

    def __call__(self, t, history):
        """Return one holding per path using observations through date t."""
        if (
            isinstance(t, (bool, np.bool_))
            or not isinstance(t, (int, np.integer))
            or not 0 <= t < self.coefficients.shape[0]
        ):
            raise ValueError("t must identify a valid trading interval.")

        if not isinstance(history, torch.Tensor):
            raise TypeError("history must be a PyTorch tensor.")

        if not history.is_floating_point():
            raise TypeError("history must have a floating-point dtype.")

        if (
            history.ndim != 3
            or history.shape[1] != t + 1
            or history.shape[2] != 2
        ):
            raise ValueError(
                "history must have shape (n_paths, t + 1, 2)."
            )

        if not torch.isfinite(history).all():
            raise ValueError("History must contain finite values.")

        a, b, c = torch.as_tensor(
            self.coefficients[t],
            dtype=history.dtype,
            device=history.device,
        )

        current_stock = history[:, -1, 0]
        current_variance = history[:, -1, 1]

        return (
            a
            + b * (current_stock / self.s0 - 1.0)
            + c * (current_variance - self.v0)
        )