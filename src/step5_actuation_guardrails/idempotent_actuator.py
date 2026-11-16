from dataclasses import dataclass
import threading


@dataclass
class LedgerAccountRecord:
    account_id: str
    status: str = "ACTIVE"  # ACTIVE | FROZEN | STEP_UP_PENDING
    last_fencing_token: int = 0


class FencedIdempotentLedger:
    """
    Thread-safe / Redis-compatible atomic ledger guard enforcing:
    1. Idempotency deduplication (f(f(x)) = f(x)) across retries.
    2. Kleppmann monotonic fencing tokens (f_t > f_last) blocking stale zombie writers.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._token_counter = 0
        self._seen_idempotency_keys: dict[str, dict] = {}
        self._accounts: dict[str, LedgerAccountRecord] = {}

    def issue_fencing_token(self) -> int:
        with self._lock:
            self._token_counter += 1
            return self._token_counter

    def apply_mutation(
        self,
        idempotency_key: str,
        account_id: str,
        new_status: str,
        fencing_token: int,
    ) -> dict[str, object]:
        with self._lock:
            if idempotency_key in self._seen_idempotency_keys:
                cached = dict(self._seen_idempotency_keys[idempotency_key])
                cached["idempotent_replay"] = True
                return cached

            rec = self._accounts.setdefault(
                account_id, LedgerAccountRecord(account_id=account_id)
            )
            if fencing_token <= rec.last_fencing_token:
                raise RuntimeError(
                    f"Stale fencing token {fencing_token} <= {rec.last_fencing_token} rejected for {account_id}"
                )

            rec.status = new_status
            rec.last_fencing_token = fencing_token
            result = {
                "committed": True,
                "idempotent_replay": False,
                "account_id": account_id,
                "status": rec.status,
                "fencing_token": fencing_token,
            }
            self._seen_idempotency_keys[idempotency_key] = dict(result)
            return result

    def get_status(self, account_id: str) -> str:
        with self._lock:
            rec = self._accounts.get(account_id)
            return rec.status if rec else "ACTIVE"
