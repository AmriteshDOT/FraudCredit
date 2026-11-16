import hashlib
import math
from typing import Dict, Tuple


class BloomFilter:
    """O(k) bit-array blocklist with P_fp ~ (1 - exp(-kn/m))^k."""

    def __init__(self, expected_items: int, fp_rate: float = 1e-4):
        self.m = max(
            64, int(-(expected_items * math.log(fp_rate)) / (math.log(2) ** 2))
        )
        self.k = max(1, int((self.m / expected_items) * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)

    def _hashes(self, item: str):
        digest = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little")
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item: str) -> None:
        for idx in self._hashes(item):
            self.bits[idx >> 3] |= 1 << (idx & 7)

    def __contains__(self, item: str) -> bool:
        return all(
            (self.bits[idx >> 3] & (1 << (idx & 7))) != 0 for idx in self._hashes(item)
        )


class HyperLogLog:
    """O(1) space cardinality estimator: E_hat = alpha_m * m^2 / sum(2^-M[j])."""

    def __init__(self, p: int = 10):
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

    def add(self, item: str) -> None:
        h = int.from_bytes(hashlib.sha256(item.encode("utf-8")).digest()[:8], "big")
        j = h >> (64 - self.p)
        w = (h << self.p) & ((1 << 64) - 1)
        rho = (64 - self.p) - w.bit_length() + 1 if w != 0 else (64 - self.p + 1)
        if rho > self.registers[j]:
            self.registers[j] = rho

    def count(self) -> float:
        harmonic_sum = sum(2.0 ** (-r) for r in self.registers)
        return self.alpha * (self.m**2) / harmonic_sum


class ContinuousEMAVelocity:
    """O(1) memory & O(1) update continuous-time exponential velocity."""

    def __init__(self, half_life_sec: float = 3600.0):
        self.lam = math.log(2.0) / half_life_sec
        self.state: Dict[str, Tuple[float, float]] = {}

    def update_and_get(self, user_id: str, amount: float, now_sec: float) -> float:
        prev_val, prev_t = self.state.get(user_id, (0.0, now_sec))
        dt = max(0.0, now_sec - prev_t)
        decayed = amount + math.exp(-self.lam * dt) * prev_val
        self.state[user_id] = (decayed, now_sec)
        return decayed
