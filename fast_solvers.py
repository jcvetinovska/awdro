"""Drop-in faster versions of RectangularSolver and RelaxedSuperhedgingSolver.

Nothing in the original files is modified.  The subclasses below build exactly
the same matrices / linear programs, but

* the setup (adversarial paths, transport costs, causality constraints) is
  vectorised with numpy instead of Python loops, and
* the two adaptive-hedge LPs are assembled directly as scipy sparse matrices
  and solved by HiGHS through scipy.optimize.linprog, instead of building
  one cvxpy expression per path / per history.

Usage in the notebook (everything else stays the same):

    from fast_solvers import FastRectangularSolver, FastRelaxedSuperhedgingSolver
    rectangular = FastRectangularSolver(model, times, BUDGET, discretization)
    outer = FastRelaxedSuperhedgingSolver(problem=problem,
                                          inner_solver=nonrectangular,
                                          penalty=PENALTY)
"""

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack, identity

from RectangularSolver import RectangularSolver
from RelaxedSuperhedgingSolver import RelaxedSuperhedgingSolver


# --------------------------------------------------------------------------
# Setup: same outputs as RectangularSolver, without Python loops
# --------------------------------------------------------------------------
class FastRectangularSolver(RectangularSolver):

    def _build_adversarial_paths(self):
        initial_state = self.reference_nodes[0]["state"]
        candidate_sets = [
            self.adversarial_candidates[t]
            for t in range(1, len(self.trading_times))
        ]
        sizes = [len(c) for c in candidate_sets]
        m = int(np.prod(sizes))

        # C-order index grid: last date varies fastest, like itertools.product.
        index = np.indices(sizes).reshape(len(sizes), -1)

        paths = np.empty((m, len(sizes) + 1, 2), dtype=float)
        paths[:, 0, :] = initial_state
        for k, candidates in enumerate(candidate_sets):
            paths[:, k + 1, :] = candidates[index[k]]
        return paths

    def _build_transport_costs(self):
        reference = self.reference_paths[:, 1:, :]
        adversarial = self.adversarial_paths[:, 1:, :]
        costs = np.empty((reference.shape[0], adversarial.shape[0]))
        for i in range(reference.shape[0]):
            diff = adversarial - reference[i][None, :, :]
            costs[i] = np.einsum("jtk,jtk->j", diff, diff)
        return costs

    def _build_causality_constraints(self):
        n = self.reference_paths.shape[0]
        m = self.adversarial_paths.shape[0]
        probabilities = self.reference_probabilities

        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
            raise ValueError(
                "Reference-path probabilities must be finite and positive."
            )

        rows, cols, vals = [], [], []
        next_row = 0
        column_offsets = np.arange(m)

        for t in range(1, len(self.trading_times) - 1):
            reference_prefixes = self.reference_paths[:, :t + 1, :].reshape(n, -1)
            adversarial_prefixes = self.adversarial_paths[:, :t + 1, :].reshape(m, -1)

            _, reference_groups = np.unique(
                reference_prefixes, axis=0, return_inverse=True
            )
            _, adversarial_groups = np.unique(
                adversarial_prefixes, axis=0, return_inverse=True
            )
            reference_groups = np.asarray(reference_groups).ravel()
            adversarial_groups = np.asarray(adversarial_groups).ravel()
            n_adv_groups = int(adversarial_groups.max()) + 1

            members, anchors = [], []
            for group in range(int(reference_groups.max()) + 1):
                indices = np.flatnonzero(reference_groups == group)
                members.extend(indices[1:].tolist())
                anchors.extend([int(indices[0])] * (len(indices) - 1))

            members = np.asarray(members, dtype=int)
            anchors = np.asarray(anchors, dtype=int)
            if members.size == 0:
                continue

            pair = np.arange(members.size)
            row_block = (
                next_row
                + pair[:, None] * n_adv_groups
                + adversarial_groups[None, :]
            )

            rows += [row_block.ravel(), row_block.ravel()]
            cols += [
                (members[:, None] * m + column_offsets[None, :]).ravel(),
                (anchors[:, None] * m + column_offsets[None, :]).ravel(),
            ]
            vals += [
                np.repeat(1.0 / probabilities[members], m),
                np.repeat(-1.0 / probabilities[anchors], m),
            ]
            next_row += members.size * n_adv_groups

        if rows:
            A_eq = coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(next_row, n * m),
                dtype=float,
            ).tocsr()
        else:
            A_eq = csr_matrix((0, n * m), dtype=float)

        return A_eq, np.zeros(next_row, dtype=float)


# --------------------------------------------------------------------------
# Outer problems: same LPs, assembled as sparse matrices, solved by HiGHS
# --------------------------------------------------------------------------
def _solve_lp(c, A_ub, b_ub, lower, upper, label):
    bounds = np.column_stack([lower, upper])
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"{label} LP failed: {result.message}")
    return result


class FastRelaxedSuperhedgingSolver(RelaxedSuperhedgingSolver):

    # Refuse LPs with more constraint rows than this, so an infeasible grid
    # fails immediately with a clear message instead of hanging or exhausting
    # memory.  Raise it on a large machine.
    max_lp_rows = 5_000_000

    def _check_size(self, rows, label):
        if rows > self.max_lp_rows:
            raise RuntimeError(
                f"{label} LP would have {rows:,} rows "
                f"(limit {self.max_lp_rows:,}); skipped."
            )

    # ---- shared preparation -------------------------------------------
    def _prepare(self, transport_model, capital_bounds, holding_bounds):
        bounds = np.asarray([capital_bounds, holding_bounds], dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
            raise ValueError("Provide two finite bounds for capital and holding.")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Each lower bound must be below its upper bound.")

        paths = transport_model.adversarial_paths
        n_paths = paths.shape[0]

        with torch.no_grad():
            payoff_tensor = self.problem.payoffs(
                torch.tensor(paths, dtype=torch.float64)
            )
        payoffs = payoff_tensor.detach().cpu().numpy()
        if payoffs.shape != (n_paths,) or not np.isfinite(payoffs).all():
            raise ValueError("Provide one finite payoff per adversarial path.")

        G, histories_by_time, offsets = self._build_adaptive_gains_matrix(paths)
        return bounds, paths, payoffs, G, histories_by_time, offsets

    @staticmethod
    def _result(x, nh, m, histories_by_time, offsets, value, shortfall):
        return {
            "capital": float(x[0]),
            "holdings": np.asarray(x[1:1 + nh], dtype=float).copy(),
            "histories_by_time": histories_by_time,
            "offsets": offsets,
            "value": float(value),
            "worst_case_shortfall": float(shortfall),
        }

    # ---- nonrectangular -----------------------------------------------
    def solve_adaptive_hedge_nonrectangular(
        self, transport_model, capital_bounds, holding_bounds
    ):
        bounds, paths, payoffs, G, histories_by_time, offsets = (
            self._prepare(transport_model, capital_bounds, holding_bounds)
        )
        m = paths.shape[0]
        nh = G.shape[1]
        n = transport_model.reference_paths.shape[0]
        A_eq, b_eq = transport_model.A_eq, transport_model.b_eq
        A_ub, b_ub = transport_model.A_ub, transport_model.b_ub
        R, Rb = A_eq.shape[0], A_ub.shape[0]
        self._check_size(n * m + m, "Nonrectangular")

        i_h = 1
        i_s = 1 + nh
        i_mu = i_s + m
        i_beta = i_mu + R
        n_vars = i_beta + Rb

        c = np.zeros(n_vars)
        c[0] = 1.0
        c[i_mu:i_mu + R] = self.penalty * b_eq
        c[i_beta:] = self.penalty * b_ub

        # shortfall + capital + G @ holdings >= payoff
        block_gain = hstack(
            [
                csr_matrix(-np.ones((m, 1))),
                -G,
                -identity(m, format="csr"),
                csr_matrix((m, R)),
                csr_matrix((m, Rb)),
            ],
            format="csr",
        )

        # A_eq.T @ mu + A_ub.T @ beta >= shortfall repeated for each reference path
        repeat = csr_matrix(
            (np.ones(n * m), (np.arange(n * m), np.tile(np.arange(m), n))),
            shape=(n * m, m),
        )
        block_dual = hstack(
            [csr_matrix((n * m, 1 + nh)), repeat, -A_eq.T, -A_ub.T],
            format="csr",
        )

        A = vstack([block_gain, block_dual], format="csr")
        b = np.concatenate([-payoffs, np.zeros(n * m)])

        lower = np.concatenate([
            [bounds[0, 0]], np.full(nh, bounds[1, 0]),
            np.zeros(m), np.full(R, -np.inf), np.zeros(Rb),
        ])
        upper = np.concatenate([
            [bounds[0, 1]], np.full(nh, bounds[1, 1]),
            np.full(m, np.inf), np.full(R, np.inf), np.full(Rb, np.inf),
        ])

        res = _solve_lp(c, A, b, lower, upper, "Nonrectangular")
        x = res.x
        shortfall = b_eq @ x[i_mu:i_mu + R] + b_ub @ x[i_beta:]
        return self._result(
            x, nh, m, histories_by_time, offsets, res.fun, shortfall
        )

    # ---- rectangular --------------------------------------------------
    def solve_adaptive_hedge_rectangular(
        self, transport_model, capital_bounds, holding_bounds
    ):
        bounds, paths, payoffs, G, histories_by_time, offsets = (
            self._prepare(transport_model, capital_bounds, holding_bounds)
        )
        m, n_times, _ = paths.shape
        nh = G.shape[1]
        n = transport_model.reference_paths.shape[0]
        i_s = 1 + nh

        block_gain = hstack(
            [csr_matrix(-np.ones((m, 1))), -G, -identity(m, format="csr")],
            format="csr",
        )

        if transport_model.budget == 0:
            return self._rectangular_zero_budget(
                transport_model, bounds, paths, payoffs, block_gain,
                nh, m, histories_by_time, offsets,
            )

        A_eq, b_eq = transport_model.A_eq, transport_model.b_eq
        A_ub, b_ub = transport_model.A_ub, transport_model.b_ub
        if A_ub.shape[0] != 1:
            raise NotImplementedError("Expected a single global budget row.")
        R = A_eq.shape[0]
        nm = n * m

        # Prefix groups at each date (history of length t + 1).
        group_id, n_groups = [], []
        for t in range(n_times - 1):
            prefixes = paths[:, :t + 1, :].reshape(m, -1)
            _, g = np.unique(prefixes, axis=0, return_inverse=True)
            g = np.asarray(g).ravel()
            group_id.append(g)
            n_groups.append(int(g.max()) + 1)

        # Number of histories (prefixes) over all dates, before building anything.
        n_prefix_total = 0
        for t in range(n_times - 1):
            prefixes = paths[:, :t + 1, :].reshape(m, -1)
            n_prefix_total += len(np.unique(prefixes, axis=0))
        self._check_size(n_prefix_total * (nm + 1), "Rectangular")

        # Every history in the candidate space is reachable when budget > 0
        # (a vanishing sliver of mass can always be sent to it), so the
        # per-history reachability LP of the original code is skipped.
        block = R + 3                       # e (R + 1), b (1), v (1)
        group_start = []
        position = i_s + m
        prefix_index = {}
        count = 0
        for t in range(n_times - 1):
            starts = []
            for g in range(n_groups[t]):
                starts.append(position)
                prefix_index[(t, g)] = count
                position += block
                count += 1
            group_start.append(np.asarray(starts))
        n_vars = position
        n_prefix = count

        def child_variable(t, j):
            """Variable holding the value of path j's prefix of length t + 2."""
            if t + 1 == n_times - 1:
                return i_s + j
            return group_start[t + 1][group_id[t + 1][j]] + R + 2

        # (nm + 1) x R block shared by every prefix: -base_eq^T
        Bt = vstack(
            [-A_eq.T, csr_matrix(b_eq.reshape(1, -1))], format="coo"
        )
        ub_column = np.concatenate([-A_ub.toarray().ravel(), [b_ub[0]]])
        ub_rows = np.flatnonzero(ub_column)
        ub_vals = ub_column[ub_rows]

        rows, cols, vals = [], [], []
        v_rows, v_cols, v_vals = [], [], []
        i_range = np.arange(n)[:, None] * m

        for t in range(n_times - 1):
            for g in range(n_groups[t]):
                p = prefix_index[(t, g)]
                start = group_start[t][g]
                row0 = p * (nm + 1)

                rows.append(Bt.row + row0)
                cols.append(Bt.col + start)
                vals.append(Bt.data)

                J = np.flatnonzero(group_id[t] == g)

                # normalisation dual: -1 on columns (i, j) with j in history
                norm_rows = (row0 + i_range + J[None, :]).ravel()
                rows.append(norm_rows)
                cols.append(np.full(norm_rows.size, start + R))
                vals.append(np.full(norm_rows.size, -1.0))

                # budget dual
                rows.append(ub_rows + row0)
                cols.append(np.full(ub_rows.size, start + R + 1))
                vals.append(ub_vals)

                # child values
                child_cols = np.fromiter(
                    (child_variable(t, j) for j in J), dtype=np.int64,
                    count=J.size,
                )
                rows.append(norm_rows)
                cols.append(np.tile(child_cols, n))
                vals.append(np.ones(norm_rows.size))

                # node_value >= normalisation dual:  e[R] - v <= 0
                v_rows.append(p)
                v_cols.append((start + R, start + R + 2))
                v_vals.append((1.0, -1.0))

        main_rows = n_prefix * (nm + 1)
        dual = coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(main_rows, n_vars),
        ).tocsr()

        node = coo_matrix(
            (
                np.array(v_vals).ravel(),
                (np.repeat(v_rows, 2), np.array(v_cols).ravel()),
            ),
            shape=(n_prefix, n_vars),
        ).tocsr()

        gain = hstack(
            [block_gain, csr_matrix((m, n_vars - block_gain.shape[1]))],
            format="csr",
        )

        A = vstack([gain, dual, node], format="csr")
        b = np.concatenate([-payoffs, np.zeros(main_rows + n_prefix)])

        lower = np.full(n_vars, -np.inf)
        upper = np.full(n_vars, np.inf)
        lower[0], upper[0] = bounds[0]
        lower[1:1 + nh], upper[1:1 + nh] = bounds[1]
        lower[i_s:i_s + m] = 0.0
        for t in range(n_times - 1):
            for start in group_start[t]:
                lower[start + R + 1] = 0.0          # budget dual >= 0

        root_value = group_start[0][0] + R + 2
        c = np.zeros(n_vars)
        c[0] = 1.0
        c[root_value] = self.penalty

        res = _solve_lp(c, A, b, lower, upper, "Rectangular")
        return self._result(
            res.x, nh, m, histories_by_time, offsets,
            res.fun, res.x[root_value],
        )

    def _rectangular_zero_budget(
        self, transport_model, bounds, paths, payoffs, block_gain,
        nh, m, histories_by_time, offsets,
    ):
        lookup = {path.tobytes(): j for j, path in enumerate(paths)}
        weights = np.zeros(m)
        for path, probability in zip(
            transport_model.reference_paths,
            transport_model.reference_probabilities,
        ):
            weights[lookup[np.ascontiguousarray(path).tobytes()]] += probability

        c = np.concatenate([[1.0], np.zeros(nh), self.penalty * weights])
        lower = np.concatenate([[bounds[0, 0]], np.full(nh, bounds[1, 0]), np.zeros(m)])
        upper = np.concatenate([[bounds[0, 1]], np.full(nh, bounds[1, 1]), np.full(m, np.inf)])

        res = _solve_lp(c, block_gain, -payoffs, lower, upper, "Rectangular (zero budget)")
        shortfall = weights @ res.x[1 + nh:]
        return self._result(
            res.x, nh, m, histories_by_time, offsets, res.fun, shortfall
        )
