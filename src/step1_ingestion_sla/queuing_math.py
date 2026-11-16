import math


def mm1_sla_breach_prob(lam: float, mu: float, t_max_sec: float) -> float:
    """P(W > T_max) = rho * exp(-mu * (1 - rho) * T_max)."""
    rho = lam / mu
    if rho >= 1.0:
        return 1.0
    return rho * math.exp(-mu * (1.0 - rho) * t_max_sec)


def max_safe_arrival_rate(mu: float, t_max_sec: float, eps: float = 1e-4) -> float:
    """Solves rho * exp(-mu * (1 - rho) * T_max) = eps via bisection for lambda_max."""
    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mid * math.exp(-mu * (1.0 - mid) * t_max_sec) <= eps:
            lo = mid
        else:
            hi = mid
    return lo * mu


def kingman_ggc_wait_ms(lam: float, mu: float, c: int, ca2: float, cs2: float) -> float:
    """Kingman's VUT approximation for multi-server G/G/c expected queue wait (ms)."""
    rho = lam / (c * mu)
    if rho >= 1.0:
        return float("inf")
    util_factor = (rho ** (math.sqrt(2.0 * (c + 1.0)) - 1.0)) / (c * (1.0 - rho))
    var_factor = 0.5 * (ca2 + cs2)
    return util_factor * var_factor * (1000.0 / mu)
