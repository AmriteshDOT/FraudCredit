from collections import deque
import random
import time


class SlidingWindowCircuitBreaker:
    def __init__(
        self,
        window_size: int = 100,
        soft_sla_ms: float = 40.0,
        trip_thresh: float = 0.05,
    ):
        self.window = deque(maxlen=window_size)
        self.soft_sla_ms = soft_sla_ms
        self.trip_thresh = trip_thresh
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self.opened_at = 0.0
        self.attempt = 0

    def record(self, latency_ms: float, error: bool) -> None:
        breached = 1 if (latency_ms > self.soft_sla_ms or error) else 0
        self.window.append(breached)
        if (
            len(self.window) >= 20
            and (sum(self.window) / len(self.window)) > self.trip_thresh
        ):
            if self.state != "OPEN":
                self.state = "OPEN"
                self.opened_at = time.monotonic()
                self.attempt += 1

    def allow_ml_call(self) -> bool:
        if self.state == "CLOSED":
            return True
        backoff = random.uniform(0.0, min(2.0, 0.05 * (2 ** min(self.attempt, 6))))
        if (time.monotonic() - self.opened_at) >= backoff:
            self.state = "HALF_OPEN"
            return True
        return False
