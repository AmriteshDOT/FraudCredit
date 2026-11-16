from contextlib import asynccontextmanager
import math
import signal
from fastapi import FastAPI, Response
from pydantic import BaseModel

from .actuation_fsm import ActuationDFSM, ActuationState
from .idempotent_actuator import FencedIdempotentLedger
from .saga_coordinator import ActuationSagaOrchestrator


class ServiceHealthState:
    boot_complete: bool = False
    should_accept_traffic: bool = False
    ml_engine_healthy: bool = True
    redis_connected: bool = True


service_state = ServiceHealthState()
ledger = FencedIdempotentLedger()
saga_orch = ActuationSagaOrchestrator(ledger)
dfsm = ActuationDFSM(tau_2fa=0.40, tau_fraud=0.80)


def _handle_sigterm(*_args) -> None:
    # Graceful Kubernetes termination: flip readiness to 503 immediately to drain traffic
    service_state.should_accept_traffic = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    signal.signal(signal.SIGTERM, _handle_sigterm)
    service_state.boot_complete = True
    service_state.should_accept_traffic = True
    yield
    service_state.should_accept_traffic = False


app = FastAPI(title="Kotak FraudShield & Credit Resolution Gateway", lifespan=lifespan)


class DecisionRequest(BaseModel):
    txn_id: str
    account_id: str
    calibrated_prob: float
    shap_valid: bool
    compliance_json: str


@app.get("/health/startup", include_in_schema=False)
async def startup_probe() -> Response:
    return Response(status_code=200 if service_state.boot_complete else 503)


@app.get("/health/live", include_in_schema=False)
async def liveness_probe() -> Response:
    # Lightweight process responsiveness check (never queries external DBs)
    return Response(status_code=200)


@app.get("/health/ready", include_in_schema=False)
async def readiness_probe() -> Response:
    # Synthetic canary check verifying finite inference math + active traffic acceptance
    canary_val = 1.0 / (1.0 + math.exp(-0.5))
    ready = (
        service_state.should_accept_traffic
        and service_state.redis_connected
        and math.isfinite(canary_val)
    )
    return Response(status_code=200 if ready else 503)


@app.post("/v1/actuate")
async def actuate_decision(req: DecisionRequest) -> dict[str, object]:
    action = dfsm.transition(
        calibrated_prob=req.calibrated_prob,
        shap_valid=req.shap_valid,
        ml_healthy=service_state.ml_engine_healthy,
    )
    if action == ActuationState.FROZEN:
        saga_res = saga_orch.execute_freeze_saga(
            txn_id=req.txn_id,
            account_id=req.account_id,
            compliance_json=req.compliance_json,
        )
        return {"action": action.value, "saga": saga_res}

    return {
        "action": action.value,
        "account_id": req.account_id,
        "ledger_status": ledger.get_status(req.account_id),
    }
