import numpy as np
import math

class Heston:
    """Represent a Heston model with constant parameters.

        Parameters
        ----------
        s0 : float
            Initial stock price. Must be positive
        v0 : float
            Initial variance, not volatility. Must be nonnegative
        mu : float
            Stock-price drift.
        kappa : float
            Positive mean-reversion speed of the variance process
        theta : float
            Nonnegative long-run variance level
        vol_of_vol : float
            Nonnegative diffusion coefficient of the variance process
            Volatility of volatility (also known as xi)
        rho : float
            Correlation between the stock and variance Brownian motions
            Must lie in [-1, 1].
    """
    def __init__(
        self,
        *,
        s0,
        v0,
        mu,
        kappa,
        theta,
        vol_of_vol,
        rho,
    ):
        self.s0 = float(s0)
        self.v0 = float(v0)
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.vol_of_vol = float(vol_of_vol)
        self.rho = float(rho)
        self._validate_params()

    def _validate_params(self):
        """Validate the Heston model parameters"""
        if not math.isfinite(self.s0) or self.s0 <= 0:
            raise ValueError("s0 must be a positive finite number.")

        if not math.isfinite(self.v0) or self.v0 < 0:
            raise ValueError("v0 must be a nonnegative finite number.")

        if not math.isfinite(self.mu):
            raise ValueError("mu must be a finite number.")

        if not math.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be a positive finite number.")

        if not math.isfinite(self.theta) or self.theta < 0:
            raise ValueError("theta must be a nonnegative finite number.")

        if not math.isfinite(self.vol_of_vol) or self.vol_of_vol < 0:
            raise ValueError(
                "vol_of_vol must be a nonnegative finite number.")
        if not math.isfinite(self.rho) or not -1 <= self.rho <= 1:
            raise ValueError("rho must be finite and between -1 and 1.")
       
            

    def simulate_paths(self, n_paths, n_steps, horizon, seed=None):  
        """ Simulate approximate Heston paths with constant parameters.
        
        Parameters
        ----------
        n_paths: int
            Number of independent paths
        n_steps: int
            Number of simulation intervals
        horizon:float
            Simulation horizon in years
        seed: int
            Seed for reproducible random draws

        Returns
        -------
        numpy.ndarray
            Array of shape (n_paths, n_steps +1, 2)
            The last axis of size two contains stock price and variance respectively
        
        """
        if not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
            raise ValueError("n_paths must be a positive integer.")
        if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
            raise ValueError("n_steps must be a positive integer.")
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizon must be finite and positive.")

        dt = horizon / n_steps
        rng = np.random.default_rng(seed)

        # Store both state variables, including their initial values.
        paths = np.empty((n_paths, n_steps + 1, 2))
        paths[:, 0, 0] = self.s0
        paths[:, 0, 1] = self.v0

        for t in range(n_steps):
            paths[:, t + 1, :] = self.step(paths[:, t, :], dt, rng)

        return paths

    def step(self, state, dt, rng):
        """ Generate next timestep Heston stock price and variance

        Parameters
        ----------
        state : numpy.ndarray
            Array of shape (n_paths, 2), containing stock prices
            in column 0 and variances in column 1
        dt : float
            Positive time increment in years
        rng : numpy.random.Generator
            Random-number generator shared across simulation steps

        Returns
        -------
        numpy.ndarray
            Updated states with shape (n_paths, 2).
            The input array is left unchanged.
        """
        state = np.asarray(state, dtype=float)

        if state.ndim != 2 or state.shape[1] != 2:
            raise ValueError("state must have shape (n_paths, 2).")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if not np.all(np.isfinite(state)):
            raise ValueError("state must contain only finite values.")
        if np.any(state[:, 0] <= 0) or np.any(state[:, 1] < 0):
            raise ValueError(
                "Stock prices must be positive and variances nonnegative."
            )

        s = state[:, 0]
        v = state[:, 1]
        xi = self.vol_of_vol

        decay = np.exp(-self.kappa * dt)
        one_minus_decay = -np.expm1(-self.kappa * dt)

        if xi == 0.0:
            # Deterministic variance: endpoint and integral are exact.
            v_next = self.theta + (v - self.theta) * decay
            integrated_variance = (
                self.theta * dt
                + (v - self.theta) * one_minus_decay / self.kappa
            )
            integrated_variance = np.maximum(integrated_variance, 0.0)

            z = rng.standard_normal(v.shape)
            log_return = (
                self.mu * dt
                - 0.5 * integrated_variance
                + np.sqrt(integrated_variance) * z
            )

        else:
            # Exact CIR variance transition.
            c = xi**2 * one_minus_decay / (4.0 * self.kappa)
            degrees = 4.0 * self.kappa * self.theta / xi**2
            noncentrality = v * decay / c

            poisson_draws = rng.poisson(noncentrality / 2.0)
            gamma_shape = degrees / 2.0 + poisson_draws

            # A zero shape corresponds to a point mass at zero.
            v_next = np.zeros_like(v)
            positive = gamma_shape > 0.0
            v_next[positive] = c * rng.gamma(
                shape=gamma_shape[positive],
                scale=2.0,
            )

            # Trapezoidal approximation of integrated variance.
            integrated_variance = 0.5 * (v + v_next) * dt

            z = rng.standard_normal(v.shape)
            log_return = (
                self.mu * dt
                + (self.rho * self.kappa / xi - 0.5)
                * integrated_variance
                + (self.rho / xi)
                * (v_next - v - self.kappa * self.theta * dt)
                + np.sqrt((1.0 - self.rho**2) * integrated_variance) * z
            )

        next_state = np.empty_like(state)
        next_state[:, 0] = s * np.exp(log_return)
        next_state[:, 1] = v_next

        return next_state

       


    
        





    
        
        

