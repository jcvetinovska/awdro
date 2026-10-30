import numpy as np
from scipy.optimize import minimize_scalar
import math

class RelaxedSuperhedgingSolver():


    def __init__(
        self,
        problem,
        inner_solver,
        penalty     
    ):

        self.problem = problem
        self.inner_solver = inner_solver
        self.penalty = penalty

    def objective(self,x,H):
        loss_fn = lambda paths: self.problem.shortfall(paths, x, H)
        result = self.inner_solver.solve(loss_fn)
        worst_case_shortfall = result["value"]

        return x + self.penalty * worst_case_shortfall

    def solve_capital(self,H,capital_bounds):
        lower, upper = capital_bounds
        if lower >= upper:
            raise ValueError("Lower capital bound must be less than upper bound")
        if not np.isfinite(capital_bounds).all():
            raise ValueError("Capital bounds must be finite")

        result = minimize_scalar(
            self.objective,
            args=(H,),
            bounds=(lower, upper),
            method="bounded",
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Capital optimization failed: {result.message}")
        capitals = [lower, float(result.x), upper]

        objective_values = [
            self.objective(lower, H),
            float(result.fun),
            self.objective(upper, H),
        ]
        
        if not np.isfinite(objective_values).all():
            raise ValueError("Objective values must be finite.")

        best_index = int(np.argmin(objective_values))

        return {
            "capital": float(capitals[best_index]),
            "value": float(objective_values[best_index]),
        }

    def constant_hedge(self, holding):










