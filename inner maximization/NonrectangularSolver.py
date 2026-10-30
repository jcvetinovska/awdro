import math
import numpy as np
import torch
import random

class NonrectangularSolver():

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

        self._validate_inputs()

        self.state_space = reference_model.is_valid_state
        self.reference_nodes = self._build_reference_model()


        

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


    def conditional_expectation(
        self, integrand, next_states, probabilities, child_ids
    ):
        """Average continuation values over a node's reference successors.

        Parameters
        ----------
        integrand : callable
            Called as integrand(child_id, state), returning a scalar value.
        next_states : array-like, shape (n_branches, 2)
            Successor stock prices and variances.
        probabilities : array-like, shape (n_branches,)
            Conditional probabilities of the successors.
        child_ids : array-like, shape (n_branches,)
            Successor IDs in self.reference_nodes, ordered consistently
            with next_states and probabilities.

        Returns
        -------
        float
            Conditional probability-weighted continuation value.
        """
        next_states = np.asarray(next_states, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        child_ids = np.asarray(child_ids)

        if (
            next_states.ndim != 2
            or next_states.shape[1] != 2
            or next_states.shape[0] == 0
        ):
            raise ValueError(
                "next_states must have shape (n_branches, 2), "
                "with at least one branch."
            )

        n_branches = next_states.shape[0]

        if probabilities.shape != (n_branches,):
            raise ValueError("Please give one probability per next state.")

        if child_ids.shape != (n_branches,):
            raise ValueError("Please give one child ID per next state.")

        if not np.issubdtype(child_ids.dtype, np.integer):
            raise TypeError("Child IDs must be integers.")

        if np.any(child_ids < 0) or np.any(
            child_ids >= len(self.reference_nodes)
        ):
            raise ValueError("Child IDs must identify existing reference nodes.")

        if not callable(integrand):
            raise TypeError("integrand must be callable.")

        if not np.isfinite(next_states).all():
            raise ValueError("Next states must be finite.")

        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
        ):
            raise ValueError("Probabilities must be finite and nonnegative.")

        if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("Probabilities must sum to 1.")

        total = 0.0
        for child_id, state, probability in zip(
            child_ids, next_states, probabilities
        ):
            if probability == 0:
                continue

            total += probability * integrand(int(child_id), state)

        return total
        
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
    

    def backward_value(
        self,
        t,
        reference_node_id,
        reference_history,
        adversarial_history,
        multiplier,
        loss_fn,
    ):
        if t == len(self.trading_times) - 1:
            return self.terminal_value(adversarial_history, loss_fn)
        else:
            node = self.reference_nodes[reference_node_id]
            child_ids = node["children"]
            next_states = np.array([
                self.reference_nodes[child_id]["state"]
                for child_id in child_ids
            ])
            integrand = lambda child_id, reference_next: self.maximize_next_state(
                t,
                child_id,
                reference_history,
                adversarial_history,
                reference_next,
                multiplier,
                loss_fn,
            )
            return self.conditional_expectation(integrand, next_states, node["probabilities"],child_ids)



    def maximize_next_state(    
        self,
        t,
        reference_next_node_id,
        reference_history,
        adversarial_history,
        reference_next,
        multiplier,
        loss_fn
    ):

        extended_reference = np.vstack([reference_history, reference_next])
        def objective(adversarial_next):
            extended_adversarial = np.vstack([adversarial_history, adversarial_next])
            continuation = self.backward_value(
                t+1,
                reference_next_node_id,
                extended_reference,
                extended_adversarial,
                multiplier,
                loss_fn
            )
            return continuation - multiplier*stage_cost(reference_next, adversarial_next)

        

        pass


    def dual_objective(self, multiplier, loss_fn):

        pass

    def solve(self, loss_fn):
        
        pass

