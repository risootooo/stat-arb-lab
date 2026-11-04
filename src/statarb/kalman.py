"""
Kalman-filter dynamic hedge ratio, implemented from scratch with numpy.

State-space model (a classic in the pairs-trading literature, e.g.
Chan, *Algorithmic Trading*, 2013):

    observation:  y_t = [1, x_t] @ theta_t + e_t,   e_t ~ N(0, R)
    state:        theta_t = theta_{t-1} + w_t,       w_t ~ N(0, Q)

with theta_t = (alpha_t, beta_t) the time-varying intercept and hedge
ratio, Q = (delta / (1 - delta)) * I  and  R = 1 by default.  A small
delta (e.g. 1e-4) makes the hedge ratio adapt slowly; larger values track
faster but chase noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanResult:
    alpha: pd.Series          # filtered intercept path
    beta: pd.Series           # filtered hedge-ratio path
    spread: pd.Series         # y_t - alpha_t - beta_t * x_t (filtered, in-sample)
    pred_error: pd.Series     # one-step-ahead prediction errors


def kalman_hedge_ratio(y: pd.Series, x: pd.Series,
                       delta: float = 1e-4, obs_var: float = 1.0) -> KalmanResult:
    """Run the Kalman filter for the dynamic regression y ~ 1 + x."""
    idx = y.index
    yv = y.to_numpy(float)
    xv = x.to_numpy(float)
    n = len(yv)
    Q = (delta / (1.0 - delta)) * np.eye(2)
    theta = np.zeros(2)
    P = np.zeros((2, 2))
    alphas, betas, errs = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        # predict (random-walk state: theta unchanged, covariance grows)
        P = P + Q
        h = np.array([1.0, xv[t]])
        # one-step prediction error
        y_hat = float(h @ theta)
        err = yv[t] - y_hat
        S = float(h @ P @ h) + obs_var
        K = (P @ h) / S            # Kalman gain
        theta = theta + K * err
        P = P - np.outer(K, h @ P)
        alphas[t], betas[t], errs[t] = theta[0], theta[1], err
    alpha = pd.Series(alphas, index=idx, name="kalman_alpha")
    beta = pd.Series(betas, index=idx, name="kalman_beta")
    spread = (y - (alpha + beta * x)).rename("kalman_spread")
    return KalmanResult(alpha=alpha, beta=beta, spread=spread,
                        pred_error=pd.Series(errs, index=idx, name="pred_error"))
