import numpy as np

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
        return self.payoff(paths)

    def trading_gains(self, paths, H):
        if paths.shape[1] != len(self.trading_times):
            raise ValueError(
            "trading_times must match the number of path observations."
            )
        n_paths = paths.shape[0]
        n_steps = paths.shape[1]-1
        gains = np.zeros(n_paths)
        for t in range(n_steps):
            # Only information available until the current time
            history = paths[:, :t+1, :]
            holdings = np.asarray(H(t,history), dtype = float)
            if holdings.shape != (n_paths,):
                raise ValueError("H must return shape (n_paths,).")
            if not np.all(np.isfinite(holdings)):
                raise ValueError("Holdings must be finite.")

            price_change = paths[:, t + 1, 0] - paths[:, t, 0]
            gains += holdings * price_change
        return gains
        

    def terminal_wealth(self, paths, x, H):
        return x + self.trading_gains(paths,H)
    

    def shortfall(self, paths,x ,H):
        return np.maximum(self.payoffs(paths) - self.terminal_wealth(paths,x,H),0.0)
