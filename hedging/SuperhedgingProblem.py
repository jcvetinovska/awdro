import numpy as np
import torch

class SuperhedgingProblem():
    def __init__(
            self,
            *,
            payoff,
            trading_times
        ):
            self.payoff = payoff
            self.trading_times = trading_times

    def payoffs(self, paths):
        """ Return one payoff per simulated path"""
        values =  self.payoff(paths)
        if not isinstance(values, torch.Tensor):
            raise TypeError("payoff must return a PyTorch tensor.")
        if values.shape != (paths.shape[0],):
            raise ValueError("payoff must return shape (n_paths,).")

        return values

    def trading_gains(self, paths, H):
        if paths.ndim != 3 or paths.shape[2] != 2:
            raise ValueError(
                "paths must have shape (n_paths, n_steps + 1, 2)."
            )
        if paths.shape[1] != len(self.trading_times):
            raise ValueError(
                "trading_times must match the number of path observations."
            )

        n_paths = paths.shape[0]
        n_steps = paths.shape[1] - 1
        gains = paths.new_zeros(n_paths)

        for t in range(n_steps):
            history = paths[:, :t + 1, :]
            holdings = H(t, history)

            if not isinstance(holdings, torch.Tensor):
                raise TypeError("H must return a PyTorch tensor.")
            if holdings.shape != (n_paths,):
                raise ValueError("H must return shape (n_paths,).")
            if not torch.isfinite(holdings).all():
                raise ValueError("Holdings must be finite.")

            price_change = paths[:, t + 1, 0] - paths[:, t, 0]
            gains = gains + holdings * price_change

        return gains
        

    def terminal_wealth(self, paths, x, H):
        return x + self.trading_gains(paths,H)
    

    def shortfall(self, paths,x ,H):
        """Return the nonnegative terminal shortfall for each path."""
        error = self.payoffs(paths) - self.terminal_wealth(paths, x, H)
        return torch.clamp(error, min=0.0)