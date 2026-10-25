import torch

class FirstOrderSensitivitySolver():

    def __init__(
        self, 
        problem, 
        reference_paths, 
        geometry, 
        transport,
        conditional_estimator,
        budget
        ):

        self.problem = problem
        self.reference_paths = reference_paths
        self.geometry = geometry
        self.transport = transport
        self.conditional_estimator = conditional_estimator
        self.budget = budget
    

    def nominal_value(self, loss_fn):
        loss = loss_fn(self.reference_paths)
        n_paths = self.reference_paths.shape[0]
        if loss.shape != (n_paths,) :
            raise ValueError("Loss must contain one value per path")
        avg_loss = loss.mean()
        return avg_loss



    def path_gradients(self, x, H):
        pass

    def sensitivity(self, x, H):
        pass

    def solve(self, x, H, budget):
        pass

