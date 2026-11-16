import time

from .circuit_breaker import SlidingWindowCircuitBreaker
from .fallback_rules import fast_cached_rule_engine
from .queuing_math import max_safe_arrival_rate, mm1_sla_breach_prob
from .redis_state_store import BloomFilter, ContinuousEMAVelocity, HyperLogLog
from .routing_controller import ModelBasedQueueMDP, ModelFreeRouterQLearning
from .wire_codec import Transaction, varint_encode, zigzag_encode


def run_step1_simulation() -> None:
    mu_pod = 1000.0
    lam_max = max_safe_arrival_rate(mu=mu_pod, t_max_sec=0.050, eps=1e-4)
    p_breach = mm1_sla_breach_prob(lam=lam_max, mu=mu_pod, t_max_sec=0.050)
    print(
        f"[Queuing Bound] lambda_max = {lam_max:.2f} req/s | P(W > 50ms) = {p_breach:.2e}"
    )

    mdp = ModelBasedQueueMDP(max_q=15, lam=lam_max, mu_ml=mu_pod)
    mdp.solve_value_iteration()
    print(
        f"[Model-Based (S,A,P,R,V)] Policy: {{q: mdp.policy[q] for q in [0, 5, 10, 15]}}"
    )

    blocklist = BloomFilter(expected_items=100_000, fp_rate=1e-4)
    blocklist.add("acct_mule_99")
    user_ips = HyperLogLog(p=10)
    velocity_tracker = ContinuousEMAVelocity(half_life_sec=3600.0)
    cb = SlidingWindowCircuitBreaker(window_size=50, soft_sla_ms=40.0, trip_thresh=0.05)
    q_router = ModelFreeRouterQLearning()

    now = time.time()
    for step in range(30):
        txn = Transaction(
            f"tx_{step}",
            "acct_user_42",
            150_000.0,
            f"10.0.1.{step % 8}",
            now + step * 0.01,
        )
        _wire_bytes = varint_encode(zigzag_encode(int(txn.amount)))

        vel_1h = velocity_tracker.update_and_get(txn.user_id, txn.amount, txn.timestamp)
        user_ips.add(txn.ip_addr)
        is_blocked = txn.user_id in blocklist

        simulated_ml_latency_ms = 55.0 if 10 <= step <= 20 else 8.0

        if cb.allow_ml_call():
            lat_ms = simulated_ml_latency_ms
            cb.record(latency_ms=lat_ms, error=(lat_ms > 40.0))
            route_used = "DEEP_ML" if lat_ms <= 40.0 else "TIMEOUT_FALLBACK"
            risk_score = (
                0.88
                if lat_ms <= 40.0
                else fast_cached_rule_engine(
                    txn.amount, vel_1h, is_blocked, user_ips.count()
                )
            )
        else:
            lat_ms = 0.3
            route_used = "CIRCUIT_OPEN_RULES"
            risk_score = fast_cached_rule_engine(
                txn.amount, vel_1h, is_blocked, user_ips.count()
            )

        r_obs = (10.0 if route_used == "DEEP_ML" else 6.5) - 50.0 * max(
            0.0, lat_ms - 40.0
        )
        q_router.update(
            (lat_ms, cb.state == "OPEN"),
            "ROUTE_DEEP_ML",
            r_obs,
            (lat_ms, cb.state == "OPEN"),
        )

    print(
        f"[Final Runtime State] CB={cb.state} | Vel_1h={vel_1h:.1f} | "
        f"Est_IPs={user_ips.count():.1f} | Risk={risk_score:.2f}"
    )


if __name__ == "__main__":
    run_step1_simulation()
