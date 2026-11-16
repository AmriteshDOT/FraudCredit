kotak-fraudshield-credit-risk/
├── README.md                                 # Mathematical formulas, architecture diagram & quickstart commands
├── pyproject.toml                            # Package configuration & pytest setup
├── Makefile                                  # `make test` and `make run-step1` shortcuts
│
├── src/
│   └── step1_ingestion_sla/                  # Step 1: Sub-50ms Ingestion & Queuing Control Package
│       ├── __init__.py                       # Exports clean public interfaces
│       ├── wire_codec.py                     # Transaction dataclass + ZigZag & Base-128 Varint wire encoders
│       ├── redis_state_store.py              # O(1) BloomFilter, HyperLogLog & ContinuousEMAVelocity
│       ├── queuing_math.py                   # M/M/1 SLA breach prob, lambda_max solver & Kingman G/G/c
│       ├── circuit_breaker.py                # 40ms sliding-window Circuit Breaker + full-jitter backoff
│       ├── fallback_rules.py                 # Sub-millisecond deterministic fallback rule engine
│       ├── routing_controller.py             # ModelBasedQueueMDP (S,A,P,R,V) + ModelFreeRouterQLearning (s,a,r,s',Q)
│       └── pipeline.py                       # Wires all modules above into the live ingestion flow
│
└── tests/
    └── test_step1_invariants.py              # Automated mathematical invariant & failure-injection tests