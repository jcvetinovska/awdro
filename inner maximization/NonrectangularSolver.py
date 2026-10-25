import math
import numpy as np
import torch

class NonrectangularSolver():

    def __init__(
        self,
        reference_model,
        model,
        trading_times,
        budget,
        state_space,
        integrator,
        maximizer,
        dual_optimizer
    ):

        self.reference_model = reference_model
        self.trading_times = trading_times
        self.budget = budget
        self.state_space = state_space
        self.integrator = integrator
        self.maximizer = maximizer
        self.dual_optimizer = dual_optimizer

        self.state_space = model.is_valid_state


        self._validate_inputs()

def _validate_inputs(self):
    if not math.isfinite(self.budget) or self.budget < 0:
        raise ValueError("Budget must be a nonnegative finite number")

    self.trading_times = np.asarray(self.trading_times, dtype=float)

    if self.trading_times.ndim != 1 or self.trading_times.size < 2:
        raise ValueError(
            "trading_times must be one-dimensional with at least two times"
        )

    if not np.isfinite(self.trading_times).all():
        raise ValueError("Times must be finite")

    if self.trading_times[0] != 0:
        raise ValueError("Trading times must start at zero")

    if np.any(np.diff(self.trading_times) <= 0):
        raise ValueError("Trading times must be strictly increasing")

    if not callable(self.state_space):
        raise TypeError("state_space must be callable")

    initial_state = np.array([
        self.reference_model.s0,
        self.reference_model.v0,
    ])

    if not self.state_space(initial_state):
        raise ValueError(
            "Initial state is outside the admissible state space"
        )
    
    
    def stage_cost(self, reference_state,adversarial_state):
        reference_state = np.asarray(reference_state,dtype = float)
        adversarial_state = np.asarray(adversarial_state,dtype = float)

        diff = reference_state - adversarial_state
        squared_diff = diff**2
        cost = squared_diff.sum()
        return cost

    def terminal_value(self, adversarial_history, loss_fn):
        adversarial_history = torch.tensor(adversarial_history, dtype = torch.float64)
        batch = torch.unsqueeze(adversarial_history, 0)
        loss = loss_fn(batch)

        if not isinstance(loss, torch.Tensor):
            raise TypeError("Loss must be a PyTorch tensor")

        if loss.shape != (1,):
            raise ValueError("Loss must have shape (1,)")
        if not torch.isfinite(loss).all():

            raise ValueError("Loss must contain finite values")

        return loss.item()
    

    def maximize_next_state():
        pass

    def backward_value():
        pass

    def dual_objective(self, multiplier, loss_fn):
        pass

    def solve(self, loss_fn):
        pass

