from src.step1_ingestion_sla.circuit_breaker import SlidingWindowCircuitBreaker
from src.step1_ingestion_sla.queuing_math import (
    max_safe_arrival_rate,
    mm1_sla_breach_prob,
)
from src.step1_ingestion_sla.redis_state_store import (
    BloomFilter,
    ContinuousEMAVelocity,
    HyperLogLog,
)
from src.step1_ingestion_sla.routing_controller import ModelBasedQueueMDP


def test_mm1_sla_invariant():
    mu = 1000.0
    lam_max = max_safe_arrival_rate(mu=mu, t_max_sec=0.050, eps=1e-4)
    assert mm1_sla_breach_prob(lam=lam_max, mu=mu, t_max_sec=0.050) <= 1e-4


def test_bloom_and_hll_invariants():
    bf = BloomFilter(expected_items=1000, fp_rate=1e-3)
    bf.add("mule_1")
    assert "mule_1" in bf  # Zero false negatives guaranteed

    hll = HyperLogLog(p=10)
    for i in range(100):
        hll.add(f"192.168.0.{i}")
    assert 85.0 <= hll.count() <= 115.0


def test_circuit_breaker_trips_above_40ms():
    cb = SlidingWindowCircuitBreaker(window_size=25, soft_sla_ms=40.0, trip_thresh=0.05)
    for _ in range(25):
        cb.record(latency_ms=48.0, error=False)
    assert cb.state == "OPEN"


def test_mdp_switches_off_deep_ml_at_high_queue():
    mdp = ModelBasedQueueMDP(max_q=15, lam=800.0, mu_ml=1000.0)
    mdp.solve_value_iteration()
    assert mdp.policy[0] == "ROUTE_DEEP_ML"
    assert mdp.policy[15] != "ROUTE_DEEP_ML"
