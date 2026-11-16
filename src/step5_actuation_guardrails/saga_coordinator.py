from .idempotent_actuator import FencedIdempotentLedger


class ActuationSagaOrchestrator:
    """
    Orchestrated Distributed Saga across:
      Step A: Ledger Debit Freeze (with monotonic fencing token)
      Step B: Regulatory XAI Audit Log Persistence
      Step C: Customer Adverse-Action Alert Dispatch
    Executes compensating rollback (UNFREEZE -> ACTIVE) if any downstream step fails.
    """

    def __init__(self, ledger: FencedIdempotentLedger):
        self.ledger = ledger
        self.audit_log: list[dict] = []
        self.dispatched_alerts: list[dict] = []

    def execute_freeze_saga(
        self,
        txn_id: str,
        account_id: str,
        compliance_json: str,
        simulate_audit_failure: bool = False,
    ) -> dict[str, object]:
        token = self.ledger.issue_fencing_token()
        freeze_key = f"saga:{txn_id}:freeze"

        # Step A: Commit account freeze
        commit_res = self.ledger.apply_mutation(
            idempotency_key=freeze_key,
            account_id=account_id,
            new_status="FROZEN",
            fencing_token=token,
        )
        if commit_res.get("idempotent_replay"):
            return commit_res

        try:
            # Step B: Mandatory RBI XAI Audit Log persistence
            if simulate_audit_failure:
                raise IOError("Regulatory audit sink unavailable")
            self.audit_log.append(
                {"txn_id": txn_id, "account_id": account_id, "payload": compliance_json}
            )

            # Step C: Customer 2FA/Alert Notification
            self.dispatched_alerts.append(
                {"txn_id": txn_id, "account_id": account_id, "channel": "SMS_AND_APP"}
            )
            return {
                "saga_status": "COMMITTED",
                "account_id": account_id,
                "ledger_status": self.ledger.get_status(account_id),
                "fencing_token": token,
            }
        except Exception as exc:
            # Compensating transaction: semantically undo ledger freeze
            comp_token = self.ledger.issue_fencing_token()
            self.ledger.apply_mutation(
                idempotency_key=f"saga:{txn_id}:compensate_unfreeze",
                account_id=account_id,
                new_status="ACTIVE",
                fencing_token=comp_token,
            )
            return {
                "saga_status": "COMPENSATED_ROLLED_BACK",
                "reason": str(exc),
                "account_id": account_id,
                "ledger_status": self.ledger.get_status(account_id),
            }
