import math
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, kron, vstack, hstack
from itertools import product


class RectangularSolver():

    def __init__(
        self,
        reference_model,
        trading_times,
        budget,
        discretization
    ):

        self.reference_model = reference_model
        self.trading_times = trading_times
        self.budget = budget
        self.discretization = discretization

        self.state_space = reference_model.is_valid_state
        self._validate_inputs()

        self.reference_nodes = self._build_reference_model()
        self.adversarial_candidates = self._build_adversarial_candidates()

        self.reference_paths, self.reference_probabilities = (self._extract_reference_paths())
        self.adversarial_paths = self._build_adversarial_paths()

        self.transport_costs = self._build_transport_costs()

        reference_A, reference_b = self._build_reference_constraints()
        causality_A, causality_b = self._build_causality_constraints()

        self.A_eq = vstack(
            [reference_A, causality_A],
            format="csr",
        )
        self.b_eq = np.concatenate([reference_b, causality_b])

        self.A_ub, self.b_ub = self._build_budget_constraint()


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

    def nominal_value(self, reference_node_id, reference_history, loss_fn):
        node = self.reference_nodes[reference_node_id]

        if not node["children"]:
            return self.terminal_value(reference_history, loss_fn)

        total = 0.0

        for child_id, probability in zip(
            node["children"], node["probabilities"]
        ):
            child_state = self.reference_nodes[child_id]["state"]
            reference_continue = np.vstack([
                reference_history,
                child_state,
            ])

            continuation = self.nominal_value(
                child_id,
                reference_continue,
                loss_fn,
            )

            total += probability * continuation

        return total

    def _build_adversarial_candidates(self):
        candidates = {}
        for t in range(len(self.trading_times)):
            states = [
                node["state"]
                for node in self.reference_nodes
                if node["time_index"] == t
            ]
            candidates[t] = np.unique(
                np.asarray(states, dtype=float),
                axis=0,
            )
        return candidates

    def _build_reference_model(self):
        """Build a finite branching approximation of the reference process.

        Starting from the model's initial state, generate a fixed number of
        successor states from every nonterminal node using the model's
        conditional sampler, ``step(state, dt, rng)``.

        Settings
        --------
        self.discretization["seed"] : int or None
            Seed for the NumPy random-number generator.
        self.discretization["reference_branches"] : int
            Positive number of children generated for each nonterminal node.
        self.trading_times : array-like
            Validated trading dates, starting at zero and strictly increasing.

        Returns
        -------
        nodes : list of dict
            Each node's position in the list is its unique identifier.
            Node zero is the initial node. Each dictionary contains:

            state : numpy.ndarray, shape (2,)
                Stock price and variance at this node.
            time_index : int
                Index of the node's date in ``self.trading_times``.
            children : list of int
                Identifiers of the node's immediate successors.
            probabilities : numpy.ndarray
                Conditional probabilities of the children, in the same
                order as ``children``. These are not unconditional node
                probabilities.

            Terminal nodes have empty children and probability arrays.

        Notes
        -----
        Each child receives conditional probability 1 / reference_branches.
        This defines a conditional Monte Carlo approximation of the model;
        it does not give exact probabilities for the sampled continuous states.

        Build this structure once and reuse it throughout the optimization
        over the transport multiplier.

        With B branches and T intervals, the number of nodes is
        1 + B + B**2 + ... + B**T, so begin with a short time horizon.
        """
        n_child_nodes = self.discretization["reference_branches"]

        if (
            isinstance(n_child_nodes, (bool, np.bool_))
            or not isinstance(n_child_nodes, (int, np.integer))
            or n_child_nodes <= 0
        ):
            raise ValueError("reference_branches must be a positive integer.")

        rng = np.random.default_rng(self.discretization["seed"])

        initial_state = np.asarray(
            [self.reference_model.s0, self.reference_model.v0],
            dtype=float,
        )

        # A node's index in this list is its ID. The root therefore has ID 0.
        nodes = [{
            "state": initial_state,
            "time_index": 0,
            "children": [],
            "probabilities": np.array([], dtype=float),
        }]

        # Track the IDs at the current date, rather than looping over a list
        # that we are simultaneously extending.
        current_level = [0]

        for t in range(len(self.trading_times) - 1):
            dt = self.trading_times[t + 1] - self.trading_times[t]
            next_level = []

            for parent_id in current_level:
                parent = nodes[parent_id]

                # Every child starts from the same parent state. The model
                # supplies independent conditional draws for these rows.
                starting_states = np.tile(
                    parent["state"],
                    (n_child_nodes, 1),
                )
                next_states = np.asarray(
                    self.reference_model.step(starting_states, dt, rng),
                    dtype=float,
                )

                if next_states.shape != (n_child_nodes, 2):
                    raise ValueError(
                        "reference_model.step must return shape "
                        "(reference_branches, 2)."
                    )

                if not np.isfinite(next_states).all():
                    raise ValueError(
                        "The reference model generated nonfinite states."
                    )

                if not all(self.state_space(state) for state in next_states):
                    raise ValueError(
                        "The reference model generated an inadmissible state."
                    )

                # These probabilities describe transitions from this parent.
                parent["probabilities"] = np.full(
                    n_child_nodes,
                    1.0 / n_child_nodes,
                )

                for state in next_states:
                    child_id = len(nodes)
                    parent["children"].append(child_id)

                    nodes.append({
                        "state": state.copy(),
                        "time_index": t + 1,
                        "children": [],
                        "probabilities": np.array([], dtype=float),
                    })

                    # These children will become parents on the next iteration.
                    next_level.append(child_id)

            current_level = next_level

        return nodes

    def _extract_reference_paths(self):
        """Extract complete reference paths and their probabilities.

        Returns
        -------
        paths : numpy.ndarray
            Shape (n_reference_paths, n_times, 2).
        probabilities : numpy.ndarray
            Unconditional probability of each reference path.
        """
        parents = {}
        node_probabilities = {0: 1.0}

        # Find each child's parent and its probability of being reached.
        for parent_id, node in enumerate(self.reference_nodes):
            for child_id, probability in zip(
                node["children"], node["probabilities"]
            ):
                parents[child_id] = parent_id
                node_probabilities[child_id] = (
                    node_probabilities[parent_id] * probability
                )

        paths = []
        probabilities = []

        for node_id, node in enumerate(self.reference_nodes):
            # Only terminal nodes identify complete paths.
            if node["children"]:
                continue

            history = []
            current_id = node_id

            # Walk backwards from the terminal node to the root.
            while True:
                history.append(self.reference_nodes[current_id]["state"])

                if current_id == 0:
                    break

                current_id = parents[current_id]

            # Store observations in chronological order.
            paths.append(history[::-1])
            probabilities.append(node_probabilities[node_id])

        return (
            np.asarray(paths, dtype=float),
            np.asarray(probabilities, dtype=float),
        )

    def _build_adversarial_paths(self):
        """Enumerate paths formed from the date-specific candidate states.

        Returns
        -------
        numpy.ndarray
            Shape (n_adversarial_paths, n_times, 2).
            Every path starts at the fixed reference initial state.

        Notes
        -----
        This enumerates possible paths only. It does not assign probabilities
        or determine which probability laws belong to the ambiguity set.
        """
        initial_state = self.reference_nodes[0]["state"]

        # Exclude time zero because the initial state is fixed.
        candidate_sets = [
            self.adversarial_candidates[t]
            for t in range(1, len(self.trading_times))
        ]

        paths = []

        # Each combination contains one candidate state per future date.
        for combination in product(*candidate_sets):
            path = np.vstack([initial_state, *combination])
            paths.append(path)

        return np.asarray(paths, dtype=float)

    def _build_transport_costs(self):
        """Return squared path distances for all reference–adversarial pairs."""
        n_reference = self.reference_paths.shape[0]
        n_adversarial = self.adversarial_paths.shape[0]

        costs = np.empty((n_reference, n_adversarial))

        for i, reference_path in enumerate(self.reference_paths):
            for j, adversarial_path in enumerate(self.adversarial_paths):
                total = 0.0

                # Both paths have the same fixed initial state.
                for t in range(1, len(self.trading_times)):
                    total += self.stage_cost(
                        reference_path[t],
                        adversarial_path[t],
                    )

                costs[i, j] = total

        return costs

    def _build_reference_constraints(self):
        n = self.reference_paths.shape[0]
        m = self.adversarial_paths.shape[0]

        # Each reference path has one block of m joint probabilities.
        A_eq = kron(
            eye(n, format="csr"),
            csr_matrix(np.ones((1, m))),
            format="csr",
        )

        b_eq = self.reference_probabilities.copy()

        return A_eq, b_eq


    def _build_causality_constraints(self):
        n = self.reference_paths.shape[0]
        m = self.adversarial_paths.shape[0]

        probabilities = self.reference_probabilities

        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
            raise ValueError("Reference-path probabilities must be finite and positive.")

        row_indices = []
        column_indices = []
        coefficients = []
        constraint_id = 0

        for t in range(1, len(self.trading_times) - 1):
            reference_prefixes = self.reference_paths[:, :t + 1, :].reshape(n, -1)
            adversarial_prefixes = self.adversarial_paths[:, :t + 1, :].reshape(m, -1)

            _, reference_groups = np.unique(
                reference_prefixes,
                axis=0,
                return_inverse=True,
            )
            _, adversarial_groups = np.unique(
                adversarial_prefixes,
                axis=0,
                return_inverse=True,
            )

            # Find the members once per date, rather than repeatedly
            # inside the reference-path comparisons.
            reference_members = [
                np.flatnonzero(reference_groups == group)
                for group in np.unique(reference_groups)
            ]
            adversarial_members = [
                np.flatnonzero(adversarial_groups == group)
                for group in np.unique(adversarial_groups)
            ]

            for indices in reference_members:
                anchor = indices[0]

                for i in indices[1:]:
                    for target_indices in adversarial_members:
                        count = target_indices.size

                        # Positive coefficients for reference path i.
                        row_indices.extend([constraint_id] * count)
                        column_indices.extend((i * m + target_indices).tolist())
                        coefficients.extend([1.0 / probabilities[i]] * count)

                        # Negative coefficients for the anchor path.
                        row_indices.extend([constraint_id] * count)
                        column_indices.extend(
                            (anchor * m + target_indices).tolist()
                        )
                        coefficients.extend(
                            [-1.0 / probabilities[anchor]] * count
                        )

                        constraint_id += 1

        A_eq = coo_matrix(
            (coefficients, (row_indices, column_indices)),
            shape=(constraint_id, n * m),
            dtype=float,
        ).tocsr()

        b_eq = np.zeros(constraint_id, dtype=float)

        return A_eq, b_eq

    def _build_budget_constraint(self):
        """Build the original global expected transport-cost constraint.

        Returns
        -------
        A_ub : scipy.sparse.csr_matrix
            One row containing the flattened squared path costs.
        b_ub : numpy.ndarray
            Shape (1,), containing the squared Wasserstein radius.
        """
        A_ub = csr_matrix(self.transport_costs.reshape(1, -1))
        b_ub = np.array([self.budget**2], dtype=float)

        return A_ub, b_ub


    def conditional_worst_case(self, adversarial_history, continuation_values):
        """Compute the worst attainable conditional continuation value.

        Parameters
        ----------
        adversarial_history : array-like
            Shape (t + 1, 2), containing the observed adversarial history.
            The history must be nonterminal.
        continuation_values : array-like
            One finite value per candidate state at time t + 1, ordered
            as self.adversarial_candidates[t + 1].

        Returns
        -------
        float or None
            Worst conditional continuation value. Returns None if the LP
            reports that the history cannot have positive probability
            under any model in the original ambiguity set.

        Notes
        -----
        Uses the Charnes–Cooper transformation of a linear-fractional
        problem into an LP. This transformation introduces no additional
        approximation; the numerical solver still uses tolerances.
        """
        history = np.asarray(adversarial_history, dtype=float)
        values = np.asarray(continuation_values, dtype=float)

        if (
            history.ndim != 2
            or history.shape[1] != 2
            or not 1 <= history.shape[0] < len(self.trading_times)
        ):
            raise ValueError(
                "Provide a nonterminal history with shape (n_observations, 2)."
            )

        if not np.isfinite(history).all():
            raise ValueError("History must contain finite values.")

        t = history.shape[0] - 1
        next_states = self.adversarial_candidates[t + 1]

        if values.shape != (len(next_states),):
            raise ValueError(
                "Provide one continuation value per next-state candidate."
            )

        if not np.isfinite(values).all():
            raise ValueError("Continuation values must be finite.")

        # Identify complete adversarial paths passing through this history.
        history_mask = np.all(
            self.adversarial_paths[:, :t + 1, :] == history,
            axis=(1, 2),
        )

        if not history_mask.any():
            raise ValueError("History is absent from the candidate path space.")

        n = self.reference_paths.shape[0]
        m = self.adversarial_paths.shape[0]

        # Each matching path receives the continuation value associated
        # with its next state. Paths outside the history contribute zero.
        path_values = np.zeros(m)

        for k, state in enumerate(next_states):
            next_state_mask = np.all(
                self.adversarial_paths[:, t + 1, :] == state,
                axis=1,
            )
            path_values[history_mask & next_state_mask] = values[k]

        # Repeat the target-path coefficients for every reference path.
        # These use the same flattened ordering as the coupling variables.
        history_coefficients = np.tile(history_mask.astype(float), n)
        value_coefficients = np.tile(path_values, n)

        # Original objective:
        #     (value_coefficients @ pi) / (history_coefficients @ pi)
        #
        # Let tau = 1 / P(history) and z = tau * pi.
        # The new LP variables are [z, tau].

        # Original equality constraints become A_eq @ z - b_eq * tau = 0.
        transformed_A_eq = hstack(
            [
                self.A_eq,
                csr_matrix(-self.b_eq.reshape(-1, 1)),
            ],
            format="csr",
        )

        # Normalize the scaled probability of the history to one.
        normalization_row = csr_matrix(
            np.concatenate([history_coefficients, [0.0]]).reshape(1, -1)
        )

        transformed_A_eq = vstack(
            [transformed_A_eq, normalization_row],
            format="csr",
        )

        transformed_b_eq = np.concatenate([
            np.zeros(self.b_eq.size),
            [1.0],
        ])

        # The global budget becomes A_ub @ z - b_ub * tau <= 0.
        transformed_A_ub = hstack(
            [
                self.A_ub,
                csr_matrix(-self.b_ub.reshape(-1, 1)),
            ],
            format="csr",
        )
        transformed_b_ub = np.zeros(self.b_ub.size)

        # Negate the objective because linprog minimizes.
        objective = np.concatenate([-value_coefficients, [0.0]])

        result = linprog(
            c=objective,
            A_ub=transformed_A_ub,
            b_ub=transformed_b_ub,
            A_eq=transformed_A_eq,
            b_eq=transformed_b_eq,
            bounds=(0.0, None),
            method="highs",
        )

        if result.status == 2:
            # No feasible coupling gives this history positive probability.
            return None

        if not result.success:
            raise RuntimeError(
                f"Conditional optimization failed: {result.message}"
            )

        value = -float(result.fun)

        if not np.isfinite(value):
            raise RuntimeError("Conditional optimization returned a nonfinite value.")

        return value

    def backward_value(self, adversarial_history, loss_fn):
        history = np.asarray(adversarial_history)
        t = history.shape[0] - 1
        if t == len(self.trading_times) - 1:
            return self.terminal_value(history, loss_fn)
        else:
            next_states = self.adversarial_candidates[t+1]
            continuation_values = np.zeros(next_states.shape[0])

            for k, next_state in enumerate(next_states):
                extended_history = np.vstack([history, next_state])
                value = self.backward_value(extended_history, loss_fn)

                if value is not None:
                    continuation_values[k] = value

            return self.conditional_worst_case(history, continuation_values)

    def solve(self, loss_fn):
        initial_state = self.reference_nodes[0]["state"]
        initial_history = initial_state.reshape((1,2))

        if self.budget == 0:
            value = self.nominal_value(0, initial_history, loss_fn)

        else:
            value = self.backward_value(initial_history, loss_fn)

        if value is None:
            raise RuntimeError ("Initial history should be reachable")
        if not np.isfinite(value):
            raise RuntimeError ("Value function infinite")

        return {"value": float(value)}

