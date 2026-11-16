def fast_cached_rule_engine(
    amount: float,
    velocity_1h: float,
    is_blacklisted: bool,
    unique_ips: float,
) -> float:
    """O(1) deterministic fallback scorer (< 0.1ms) when W > 40ms."""
    if is_blacklisted:
        return 0.99
    score = 0.05
    if amount > 100_000 and velocity_1h > 250_000:
        score += 0.65
    if unique_ips > 5.0:
        score += 0.25
    return min(0.99, score)
