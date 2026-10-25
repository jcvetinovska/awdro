import torch
import math


class FirstOrderSensitivitySolver():

    def __init__(
        self,
        reference_paths,
        geometry,
        transport,
        p,
        conditional_estimator,
        budget
    ):
        self.geometry = geometry
        self.transport = transport
        self.conditional_estimator = conditional_estimator
        self.budget = budget
        self.p = p

        if not math.isfinite(self.budget) or self.budget < 0:
            raise ValueError("budget must be a nonnegative finite number")

        if not math.isfinite(self.p) or self.p <= 1:
            raise ValueError("Choose a finite p above 1")

        if geometry != "euclidean":
            raise NotImplementedError(
                "Only Euclidean distance on path levels is implemented"
            )

        if transport != "ordinary":
            raise NotImplementedError(
                "Only ordinary Wasserstein transport is implemented"
            )

        paths = torch.as_tensor(reference_paths, dtype=torch.float64)

        if paths.ndim != 3:
            raise ValueError(
                "reference_paths must have shape "
                "(n_paths, n_steps + 1, n_features)."
            )

        if paths.shape[0] == 0:
            raise ValueError("reference_paths must contain at least one path.")

        if paths.shape[1] < 2:
            raise ValueError("Each path must contain at least two observations.")

        if paths.shape[2] == 0:
            raise ValueError("Each observation must contain at least one feature.")

        if not torch.isfinite(paths).all():
            raise ValueError("reference_paths must contain only finite values.")

        self.reference_paths = paths.detach().clone()

    def nominal_value(self, loss_fn):
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable.")

        paths = self.reference_paths
        loss = loss_fn(paths)

        if not isinstance(loss, torch.Tensor):
            raise TypeError("Loss function must return a PyTorch tensor")

        if loss.shape != (paths.shape[0],):
            raise ValueError("Loss function must return shape (n_paths,)")

        if not torch.isfinite(loss).all():
            raise ValueError("Loss values must be finite.")

        avg_loss = loss.mean()
        return avg_loss

    def path_gradients(self, loss_fn):
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable.")

        paths = self.reference_paths.clone().requires_grad_(True)
        loss = loss_fn(paths)

        if not isinstance(loss, torch.Tensor):
            raise TypeError("Loss function must return a PyTorch tensor.")

        if loss.shape != (paths.shape[0],):
            raise ValueError("Loss function must return shape (n_paths,).")

        if not torch.isfinite(loss).all():
            raise ValueError("Loss values must be finite.")

        if not loss.requires_grad:
            raise ValueError(
                "Loss does not support gradient tracking. "
                "Check for NumPy operations, detach(), or a constant loss."
            )

        total_loss = loss.sum()
        gradients = torch.autograd.grad(
            outputs=total_loss,
            inputs=paths
        )[0]

        if not torch.isfinite(gradients).all():
            raise ValueError("Path gradients must be finite.")

        gradients[:, 0, :] = 0
        return gradients

    def sensitivity(self, loss_fn):
        conjugate_exp = self.p / (self.p - 1)
        gradients = self.path_gradients(loss_fn)
        sq_norm_grad = (
            torch.norm(gradients, p=2, dim=(1, 2))
        ) ** conjugate_exp
        avg = sq_norm_grad.mean()
        coefficient = avg ** (1.0 / conjugate_exp)

        if not torch.isfinite(coefficient):
            raise ValueError("Sensitivity is nonfinite.")

        return coefficient

    def solve(self, loss_fn):
        nominal_value = self.nominal_value(loss_fn)

        if self.budget == 0:
            return nominal_value
        else:
            sensitivity = self.sensitivity(loss_fn)
            approx_value = nominal_value + sensitivity * self.budget
            return approx_value

