import numpy as np
from scipy.optimize import minimize_scalar
import math
from hedging.strategies import ConstantHedge
import cvxpy as cp
import torch
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

class RelaxedSuperhedgingSolver():


    def __init__(
        self,
        problem,
        inner_solver,
        penalty     
    ):

        self.problem = problem
        self.inner_solver = inner_solver
        self.penalty = float(penalty)

        if not math.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("penalty must be finite and positive.")

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
            "value": float(objective_values[best_index])
        }

    def constant_hedge_objective(self, parameters):
        x, holding = parameters
        H = ConstantHedge(holding)
        return self.objective(x,H)


    def solve_constant_hedge_nonrectangular(
            self,
            transport_model,
            capital_bounds,
            holding_bounds,
        ):
        capital_lower, capital_upper = capital_bounds
        holding_lower, holding_upper = holding_bounds

        if not np.isfinite(capital_bounds).all():
            raise ValueError("Capital bounds must be finite.")
        if capital_lower >= capital_upper:
            raise ValueError("Lower capital bound must be less than upper bound.")

        if not np.isfinite(holding_bounds).all():
            raise ValueError("Holding bounds must be finite.")
        if holding_lower >= holding_upper:
            raise ValueError("Lower holding bound must be less than upper bound.")

        n_paths = transport_model.adversarial_paths.shape[0]

        capital = cp.Variable(name="capital")
        holding = cp.Variable(name="holding")
        shortfall = cp.Variable(
            n_paths,
            nonneg=True,
            name="shortfall",
        )
        paths = transport_model.adversarial_paths

        paths_tensor = torch.tensor(paths, dtype=torch.float64)

        with torch.no_grad():
            payoff_values = self.problem.payoffs(paths_tensor)

        payoffs = payoff_values.detach().cpu().numpy()

        if payoffs.shape != (n_paths,):
            raise ValueError("Payoff must contain one value per adversarial path.")

        if not np.isfinite(payoffs).all():
            raise ValueError("Payoffs must be finite.")

        price_changes = paths[:, -1, 0] - paths[:, 0, 0]

        constraints = [
            capital >= capital_lower,
            capital <= capital_upper,
            holding >= holding_lower,
            holding <= holding_upper,
            shortfall >= payoffs - capital - holding * price_changes,
        ]
        equality_dual = cp.Variable(
            transport_model.A_eq.shape[0],
            name="equality_dual",
        )

        budget_dual = cp.Variable(
            transport_model.A_ub.shape[0],
            nonneg=True,
            name="budget_dual",
        )

        n_reference = transport_model.reference_paths.shape[0]

        pair_shortfalls = cp.hstack(
            [shortfall for _ in range(n_reference)]
        )

        constraints.append(
            transport_model.A_eq.T @ equality_dual
            + transport_model.A_ub.T @ budget_dual
            >= pair_shortfalls
        )

        robust_shortfall_bound = (
            transport_model.b_eq @ equality_dual
            + transport_model.b_ub @ budget_dual
        )

        optimization_problem = cp.Problem(
            cp.Minimize(
                capital + self.penalty * robust_shortfall_bound
            ),
            constraints,
        )

        optimization_problem.solve(solver=cp.HIGHS)

        if optimization_problem.status != cp.OPTIMAL:
            raise RuntimeError(
                "Optimization did not report an optimal solution: "
                f"{optimization_problem.status}"
            )
        return {
            "capital": float(capital.value),
            "holding": float(holding.value),
            "value": float(optimization_problem.value),
            "worst_case_shortfall": float(robust_shortfall_bound.value),
        }

    def solve_constant_hedge_rectangular(
        self,
        transport_model,
        capital_bounds,
        holding_bounds,
    ):
        """Optimize capital and a constant holding over the finite rectangular hull."""
        bounds = np.asarray(
            [capital_bounds, holding_bounds],
            dtype=float,
        )

        if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
            raise ValueError("Provide two finite bounds for capital and holding.")

        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each lower bound must be below its upper bound.")

        if not np.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("penalty must be finite and positive.")

        paths = transport_model.adversarial_paths
        n_reference = transport_model.reference_paths.shape[0]
        n_paths, n_times, n_features = paths.shape

        with torch.no_grad():
            payoff_tensor = self.problem.payoffs(
                torch.tensor(paths, dtype=torch.float64)
            )

        if not isinstance(payoff_tensor, torch.Tensor):
            raise TypeError("Payoffs must be a PyTorch tensor.")

        payoffs = payoff_tensor.detach().cpu().numpy()

        if payoffs.shape != (n_paths,) or not np.isfinite(payoffs).all():
            raise ValueError("Provide one finite payoff per adversarial path.")

        capital = cp.Variable(name="capital")
        holding = cp.Variable(name="holding")
        shortfall = cp.Variable(
            n_paths,
            nonneg=True,
            name="shortfall",
        )

        price_changes = paths[:, -1, 0] - paths[:, 0, 0]

        constraints = [
            capital >= bounds[0, 0],
            capital <= bounds[0, 1],
            holding >= bounds[1, 0],
            holding <= bounds[1, 1],
            shortfall >= payoffs - capital - holding * price_changes,
        ]

        # Map each complete history to its terminal shortfall variable.
        values = {
            tuple(path.ravel()): shortfall[j]
            for j, path in enumerate(paths)
        }

        if transport_model.budget == 0:
            # The rectangular hull of the singleton reference law
            # has the same path distribution.
            root_value = sum(
                float(probability) * values[tuple(path.ravel())]
                for path, probability in zip(
                    transport_model.reference_paths,
                    transport_model.reference_probabilities,
                )
            )

        else:
            # Common parts of the transformed conditional primal LP.
            base_eq = hstack(
                [
                    transport_model.A_eq,
                    csr_matrix(-transport_model.b_eq.reshape(-1, 1)),
                ],
                format="csr",
            )

            local_ub = hstack(
                [
                    transport_model.A_ub,
                    csr_matrix(-transport_model.b_ub.reshape(-1, 1)),
                ],
                format="csr",
            )

            local_rhs = np.concatenate([
                np.zeros(transport_model.b_eq.size),
                [1.0],
            ])

            # Build the conditional dual constraints backwards.
            for t in range(n_times - 2, -1, -1):
                prefixes = np.unique(
                    paths[:, :t + 1, :].reshape(n_paths, -1),
                    axis=0,
                )

                for prefix in prefixes:
                    history = prefix.reshape(t + 1, n_features)

                    # Reachability depends on the original ambiguity set,
                    # not on capital, holding, or continuation values.
                    reachable = transport_model.conditional_worst_case(
                        history,
                        np.zeros(
                            len(transport_model.adversarial_candidates[t + 1])
                        ),
                    )

                    if reachable is None:
                        continue

                    history_mask = np.all(
                        paths[:, :t + 1, :] == history,
                        axis=(1, 2),
                    )

                    # Associate each complete path with its child's value.
                    path_values = []

                    for j in range(n_paths):
                        child_key = tuple(paths[j, :t + 2, :].ravel())

                        if history_mask[j]:
                            # Unreachable child histories have zero mass
                            # under every admissible parent kernel.
                            child_value = values.get(
                                child_key,
                                cp.Constant(0.0),
                            )
                        else:
                            child_value = cp.Constant(0.0)

                        path_values.append(child_value)

                    path_values = cp.hstack(path_values)

                    # Objective coefficients for [scaled coupling, tau].
                    coefficients = cp.hstack(
                        [path_values for _ in range(n_reference)]
                        + [cp.Constant(0.0)]
                    )

                    history_coefficients = np.tile(
                        history_mask.astype(float),
                        n_reference,
                    )

                    normalization_row = csr_matrix(
                        np.concatenate([
                            history_coefficients,
                            [0.0],
                        ]).reshape(1, -1)
                    )

                    local_eq = vstack(
                        [base_eq, normalization_row],
                        format="csr",
                    )

                    equality_dual = cp.Variable(local_eq.shape[0])
                    budget_dual = cp.Variable(
                        local_ub.shape[0],
                        nonneg=True,
                    )
                    node_value = cp.Variable()

                    constraints.extend([
                        local_eq.T @ equality_dual
                        + local_ub.T @ budget_dual
                        >= coefficients,

                        node_value >= local_rhs @ equality_dual,
                    ])

                    values[tuple(prefix)] = node_value

            root_key = tuple(paths[0, :1, :].ravel())

            if root_key not in values:
                raise RuntimeError("Initial history was reported unreachable.")

            root_value = values[root_key]

        optimization = cp.Problem(
            cp.Minimize(capital + self.penalty * root_value),
            constraints,
        )

        optimization.solve(solver=cp.HIGHS)

        if optimization.status != cp.OPTIMAL:
            raise RuntimeError(
                f"Rectangular optimization failed: {optimization.status}"
            )

        return {
            "capital": float(capital.value),
            "holding": float(holding.value),
            "value": float(optimization.value),
            "worst_case_shortfall": float(root_value.value),
        }

    def evaluate_constant_hedge_first_order(
        self,
        sensitivity_solver,
        capital,
        holding,
    ):
        """Evaluate the existing first-order W2 approximation at a fixed strategy."""
        capital = float(capital)
        holding = float(holding)

        if not np.isfinite([capital, holding]).all():
            raise ValueError("Capital and holding must be finite.")

        if not np.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("penalty must be finite and positive.")

        if sensitivity_solver.transport != "ordinary":
            raise NotImplementedError(
                "This method uses ordinary Wasserstein sensitivity."
            )

        if (
            sensitivity_solver.p != 2
            or sensitivity_solver.geometry != "euclidean"
        ):
            raise NotImplementedError(
                "This method requires W2 with Euclidean path geometry."
            )

        hedge = ConstantHedge(holding)

        loss_fn = lambda paths: self.problem.shortfall(
            paths,
            capital,
            hedge,
        )

        nominal = float(
            sensitivity_solver.nominal_value(loss_fn).detach().cpu()
        )

        if sensitivity_solver.budget == 0:
            coefficient = None
            approximate_shortfall = nominal
        else:
            coefficient = float(
                sensitivity_solver.sensitivity(loss_fn).detach().cpu()
            )
            approximate_shortfall = (
                nominal + sensitivity_solver.budget * coefficient
            )

        objective_value = (
            capital + self.penalty * approximate_shortfall
        )

        if not np.isfinite([
            nominal,
            approximate_shortfall,
            objective_value,
        ]).all():
            raise RuntimeError("First-order evaluation returned nonfinite values.")

        return {
            "capital": capital,
            "holding": holding,
            "value": objective_value,
            "nominal_shortfall": nominal,
            "sensitivity": coefficient,
            "approximate_worst_case_shortfall": approximate_shortfall,
        }

    def _build_history_groups(self, paths):
        paths = np.asarray(paths, dtype=float)

        n_paths, n_times, n_features = paths.shape
        n_steps = n_times - 1

        histories_by_time = []
        groups_by_time = []

        for t in range(n_steps):
            # Exclude future observations.
            histories = paths[:, :t + 1, :].reshape(n_paths, -1)

            unique_histories, groups = np.unique(
                histories,
                axis=0,
                return_inverse=True,
            )

            histories_by_time.append(
                unique_histories.reshape(-1, t + 1, n_features)
            )
            groups_by_time.append(groups)

        return histories_by_time, groups_by_time


    def _build_adaptive_gains_matrix(self, paths):
        """Build a sparse map from history-dependent holdings to path gains."""
        paths = np.asarray(paths, dtype=float)

        histories_by_time, groups_by_time = self._build_history_groups(paths)

        n_paths = paths.shape[0]
        price_changes = np.diff(paths[:, :, 0], axis=1)

        row_indices = []
        column_indices = []
        coefficients = []

        offsets = []
        offset = 0

        for t, groups in enumerate(groups_by_time):
            offsets.append(offset)

            # One coefficient per path at this trading date.
            row_indices.append(np.arange(n_paths))
            column_indices.append(offset + groups)
            coefficients.append(price_changes[:, t])

            offset += len(histories_by_time[t])

        G = coo_matrix(
            (
                np.concatenate(coefficients),
                (
                    np.concatenate(row_indices),
                    np.concatenate(column_indices),
                ),
            ),
            shape=(n_paths, offset),
        ).tocsr()

        G.eliminate_zeros()

        return G, histories_by_time, np.asarray(offsets, dtype=int)

    def solve_adaptive_hedge_nonrectangular(
        self,
        transport_model,
        capital_bounds,
        holding_bounds,
    ):

        capital_lower, capital_upper = capital_bounds
        holding_lower, holding_upper = holding_bounds

        if not np.isfinite(capital_bounds).all():
            raise ValueError("Capital bounds must be finite.")
        if capital_lower >= capital_upper:
            raise ValueError("Lower capital bound must be less than upper bound.")

        if not np.isfinite(holding_bounds).all():
            raise ValueError("Holding bounds must be finite.")
        if holding_lower >= holding_upper:
            raise ValueError("Lower holding bound must be less than upper bound.")

        paths = transport_model.adversarial_paths
        n_paths = paths.shape[0]

        G, histories_by_time, offsets = (
            self._build_adaptive_gains_matrix(paths)
        )

        capital = cp.Variable(name="capital")
        holdings = cp.Variable(G.shape[1], name="holdings")
        shortfall = cp.Variable(
            n_paths,
            nonneg=True,
            name="shortfall",
        )

        paths = transport_model.adversarial_paths

        paths_tensor = torch.tensor(paths, dtype=torch.float64)

        with torch.no_grad():
            payoff_values = self.problem.payoffs(paths_tensor)

        payoffs = payoff_values.detach().cpu().numpy()

        if payoffs.shape != (n_paths,):
            raise ValueError("Payoff must contain one value per adversarial path.")

        if not np.isfinite(payoffs).all():
            raise ValueError("Payoffs must be finite.")


        constraints = [
            capital >= capital_lower,
            capital <= capital_upper,
            holdings >= holding_lower,
            holdings <= holding_upper,
            shortfall >= payoffs - capital - G @ holdings,
        ]
        equality_dual = cp.Variable(
            transport_model.A_eq.shape[0],
            name="equality_dual",
        )

        budget_dual = cp.Variable(
            transport_model.A_ub.shape[0],
            nonneg=True,
            name="budget_dual",
        )

        n_reference = transport_model.reference_paths.shape[0]

        pair_shortfalls = cp.hstack(
            [shortfall for _ in range(n_reference)]
        )

        constraints.append(
            transport_model.A_eq.T @ equality_dual
            + transport_model.A_ub.T @ budget_dual
            >= pair_shortfalls
        )

        robust_shortfall_bound = (
            transport_model.b_eq @ equality_dual
            + transport_model.b_ub @ budget_dual
        )

        optimization_problem = cp.Problem(
            cp.Minimize(
                capital + self.penalty * robust_shortfall_bound
            ),
            constraints,
        )

        optimization_problem.solve(solver=cp.HIGHS)

        if optimization_problem.status != cp.OPTIMAL:
            raise RuntimeError(
                "Optimization did not report an optimal solution: "
                f"{optimization_problem.status}"
            )
        return {
            "capital": float(capital.value),
            "holdings": np.asarray(holdings.value, dtype=float).copy(),
            "histories_by_time": histories_by_time,
            "offsets": offsets,
            "value": float(optimization_problem.value),
            "worst_case_shortfall": float(robust_shortfall_bound.value),
        }

    def solve_adaptive_hedge_rectangular(
            self,
            transport_model,
            capital_bounds,
            holding_bounds,
        ):
        """Optimize capital and a constant holding over the finite rectangular hull."""
        bounds = np.asarray(
                [capital_bounds, holding_bounds],
                dtype=float,
            )
    
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
            raise ValueError("Provide two finite bounds for capital and holding.")

        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each lower bound must be below its upper bound.")

        if not np.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("penalty must be finite and positive.")

        paths = transport_model.adversarial_paths
        n_reference = transport_model.reference_paths.shape[0]
        n_paths, n_times, n_features = paths.shape

        with torch.no_grad():
            payoff_tensor = self.problem.payoffs(
                torch.tensor(paths, dtype=torch.float64)
            )

        if not isinstance(payoff_tensor, torch.Tensor):
            raise TypeError("Payoffs must be a PyTorch tensor.")

        payoffs = payoff_tensor.detach().cpu().numpy()

        if payoffs.shape != (n_paths,) or not np.isfinite(payoffs).all():
            raise ValueError("Provide one finite payoff per adversarial path.")

        G, histories_by_time, offsets = (
            self._build_adaptive_gains_matrix(paths)
        )

        capital = cp.Variable(name="capital")
        holdings = cp.Variable(G.shape[1], name="holdings")
        shortfall = cp.Variable(
            n_paths,
            nonneg=True,
            name="shortfall",
        )

        price_changes = paths[:, -1, 0] - paths[:, 0, 0]

        constraints = [
            capital >= bounds[0, 0],
            capital <= bounds[0, 1],
            holdings >= bounds[1, 0],
            holdings <= bounds[1, 1],
            shortfall >= payoffs - capital - G @ holdings,
        ]

        # Map each complete history to its terminal shortfall variable.
        values = {
            tuple(path.ravel()): shortfall[j]
            for j, path in enumerate(paths)
        }

        if transport_model.budget == 0:
            # The rectangular hull of the singleton reference law
            # has the same path distribution.
            root_value = sum(
                float(probability) * values[tuple(path.ravel())]
                for path, probability in zip(
                    transport_model.reference_paths,
                    transport_model.reference_probabilities,
                )
            )

        else:
            # Common parts of the transformed conditional primal LP.
            base_eq = hstack(
                [
                    transport_model.A_eq,
                    csr_matrix(-transport_model.b_eq.reshape(-1, 1)),
                ],
                format="csr",
            )

            local_ub = hstack(
                [
                    transport_model.A_ub,
                    csr_matrix(-transport_model.b_ub.reshape(-1, 1)),
                ],
                format="csr",
            )

            local_rhs = np.concatenate([
                np.zeros(transport_model.b_eq.size),
                [1.0],
            ])

            # Build the conditional dual constraints backwards.
            for t in range(n_times - 2, -1, -1):
                prefixes = np.unique(
                    paths[:, :t + 1, :].reshape(n_paths, -1),
                    axis=0,
                )

                for prefix in prefixes:
                    history = prefix.reshape(t + 1, n_features)

                    # Reachability depends on the original ambiguity set,
                    # not on capital, holding, or continuation values.
                    reachable = transport_model.conditional_worst_case(
                        history,
                        np.zeros(
                            len(transport_model.adversarial_candidates[t + 1])
                        ),
                    )

                    if reachable is None:
                        continue

                    history_mask = np.all(
                        paths[:, :t + 1, :] == history,
                        axis=(1, 2),
                    )

                    # Associate each complete path with its child's value.
                    path_values = []

                    for j in range(n_paths):
                        child_key = tuple(paths[j, :t + 2, :].ravel())

                        if history_mask[j]:
                            # Unreachable child histories have zero mass
                            # under every admissible parent kernel.
                            child_value = values.get(
                                child_key,
                                cp.Constant(0.0),
                            )
                        else:
                            child_value = cp.Constant(0.0)

                        path_values.append(child_value)

                    path_values = cp.hstack(path_values)

                    # Objective coefficients for [scaled coupling, tau].
                    coefficients = cp.hstack(
                        [path_values for _ in range(n_reference)]
                        + [cp.Constant(0.0)]
                    )

                    history_coefficients = np.tile(
                        history_mask.astype(float),
                        n_reference,
                    )

                    normalization_row = csr_matrix(
                        np.concatenate([
                            history_coefficients,
                            [0.0],
                        ]).reshape(1, -1)
                    )

                    local_eq = vstack(
                        [base_eq, normalization_row],
                        format="csr",
                    )

                    equality_dual = cp.Variable(local_eq.shape[0])
                    budget_dual = cp.Variable(
                        local_ub.shape[0],
                        nonneg=True,
                    )
                    node_value = cp.Variable()

                    constraints.extend([
                        local_eq.T @ equality_dual
                        + local_ub.T @ budget_dual
                        >= coefficients,

                        node_value >= local_rhs @ equality_dual,
                    ])

                    values[tuple(prefix)] = node_value

            root_key = tuple(paths[0, :1, :].ravel())

            if root_key not in values:
                raise RuntimeError("Initial history was reported unreachable.")

            root_value = values[root_key]

        optimization = cp.Problem(
            cp.Minimize(capital + self.penalty * root_value),
            constraints,
        )

        optimization.solve(solver=cp.HIGHS)

        if optimization.status != cp.OPTIMAL:
            raise RuntimeError(
                f"Rectangular optimization failed: {optimization.status}"
            )

        return {
            "capital": float(capital.value),
            "holdings": np.asarray(holdings.value, dtype=float).copy(),
            "histories_by_time": histories_by_time,
            "offsets": offsets,
            "value": float(optimization.value),
            "worst_case_shortfall": float(root_value.value),
        }

    def evaluate_hedge_first_order(
        self,
        sensitivity_solver,
        capital,
        H,
    ):
        """Evaluate first-order W2 sensitivity for a supplied differentiable hedge.

        H must define holdings on perturbed histories using PyTorch operations.
        An exact-history lookup table alone does not meet this requirement.
        """
        capital = float(capital)

        if not np.isfinite(capital):
            raise ValueError("Capital must be finite.")

        if not callable(H):
            raise TypeError("H must be callable.")

        if (
            sensitivity_solver.transport != "ordinary"
            or sensitivity_solver.geometry != "euclidean"
            or sensitivity_solver.p != 2
        ):
            raise NotImplementedError(
                "This evaluator requires ordinary W2 with Euclidean geometry."
            )

        loss_fn = lambda paths: self.problem.shortfall(
            paths, capital, H
        )

        nominal = float(
            sensitivity_solver.nominal_value(loss_fn).detach().cpu()
        )

        coefficient = None
        approximation = nominal

        if sensitivity_solver.budget > 0:
            coefficient = float(
                sensitivity_solver.sensitivity(loss_fn).detach().cpu()
            )
            approximation += sensitivity_solver.budget * coefficient

        value = capital + self.penalty * approximation

        if not np.isfinite([nominal, approximation, value]).all():
            raise RuntimeError("First-order evaluation returned nonfinite values.")

        return {
            "capital": capital,
            "value": value,
            "nominal_shortfall": nominal,
            "sensitivity": coefficient,
            "approximate_worst_case_shortfall": approximation,
        }




