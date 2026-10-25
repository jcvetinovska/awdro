import torch
import math
# Rewrite using JAX instead of autograd for speedup

class FirstOrderSensitivitySolver():

    def __init__(
        self,
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

        if not math.isfinite(self.budget) or self.budget < 0:
                    raise ValueError("budget must be a nonegative finite number.")
    

    def nominal_value(self, loss_fn):
        paths = torch.tensor(self.reference_paths, dtype = torch.float64)
        loss = loss_fn(paths)
        
        if not isinstance(loss, torch.tensor):
            raise TypeError("Loss function must return a PyTorch tensor.")
        if loss.shape != (paths.shape[0],):
            raise ValueError("Loss function must return shape (n_paths,).")
        
        avg_loss = loss.mean()
        return avg_loss


    def path_gradients(self, loss_fn):
        paths  = torch.tensor(self.reference_paths, dtype = torch.float64, requires_grad = True)
        loss = loss_fn(paths)
        if not isinstance(loss, torch.tensor):
            raise TypeError("Loss function must return a PyTorch tensor.")
        if loss.shape != (paths.shape[0],):
            raise ValueError("Loss function must return shape (n_paths,).")
        total_loss = loss.sum()
        gradients = torch.autograd.grad(outputs= total_loss, inputs = paths)[0]
        gradients[:, 0, :] = 0
        return gradients


    def sensitivity(self, loss_fn):
        gradients = self.path_gradients(loss_fn)
        sq_norm_grad = (torch.norm(gradients, p=2, dim = (1,2)))**2
        avg = sq_norm_grad.mean()
        return torch.sqrt(avg)

        

    def solve(self, loss_fn):
        sensitivity = self.sensitivity(loss_fn)
        nominal_value = self.nominal_value(loss_fn)
        approx_value = nominal_value + sensitivity * self.budget
        return approx_value


