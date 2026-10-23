# Payoff functions

import numpy as np


def european_call(paths, strike):
    """Pay (S_T - strike)_+."""
    s_terminal = paths[:, -1, 0]
    return np.maximum(s_terminal - strike, 0.0)


def european_put(paths, strike):
    """Pay (strike - S_T)_+."""
    s_terminal = paths[:, -1, 0]
    return np.maximum(strike - s_terminal, 0.0)


def digital_call(paths, strike, cash=1.0):
    """Pay cash if S_T > strike; otherwise pay zero."""
    s_terminal = paths[:, -1, 0]
    return cash * (s_terminal > strike).astype(float)


def digital_put(paths, strike, cash=1.0):
    """Pay cash if S_T < strike; otherwise pay zero."""
    s_terminal = paths[:, -1, 0]
    return cash * (s_terminal < strike).astype(float)


def bull_call_spread(paths, lower_strike, upper_strike):
    """Long a lower-strike call and short an upper-strike call."""
    if not lower_strike < upper_strike:
        raise ValueError("lower_strike must be below upper_strike.")

    return (
        european_call(paths, lower_strike)
        - european_call(paths, upper_strike)
    )


def bear_put_spread(paths, lower_strike, upper_strike):
    """Long an upper-strike put and short a lower-strike put."""
    if not lower_strike < upper_strike:
        raise ValueError("lower_strike must be below upper_strike.")

    return (
        european_put(paths, upper_strike)
        - european_put(paths, lower_strike)
    )


def butterfly(paths, strike, width):
    """Symmetric long call butterfly centered at strike.

    The three strikes are strike-width, strike, and strike+width.
    The corresponding call positions are +1, -2, and +1.
    """
    if not width > 0:
        raise ValueError("width must be positive.")

    return (
        european_call(paths, strike - width)
        - 2.0 * european_call(paths, strike)
        + european_call(paths, strike + width)
    )


def straddle(paths, strike):
    """Long a call and a put with the same strike; pay |S_T-strike|."""
    return (
        european_call(paths, strike)
        + european_put(paths, strike)
    )


def asian_call(paths, strike):
    """Call on the arithmetic average of post-initial stock prices.

    Every recorded date after time zero receives equal weight.
    """
    average_price = paths[:, 1:, 0].mean(axis=1)
    return np.maximum(average_price - strike, 0.0)