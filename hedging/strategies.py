import math
import torch

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
