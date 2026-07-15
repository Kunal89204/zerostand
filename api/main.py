import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from data.simulator import (
    USERS, NORMAL_USERS, DRIFT_USERS, generate_dataset,
    _normal_case, _excessive_scope_case, _peer_drift_case, _conformance_violation_case,
)
from engine.risk_scorer import RiskScorer
from engine.conformance import check_conformance
from engine.policy import decide_request, decide_execution
from pqc.tokens import TokenBroker
from pqc.ledger import AuditLedger

app = FastAPI(title="ZeroStand")

SCENARIOS = ["normal", "excessive_scope_request", "peer_drift_request", "conformance_violation", "token_replay"]

STATE = {
    "scorer": RiskScorer(),
    "broker": TokenBroker(),
    "ledger": AuditLedger(),
    "traces": [],
    "last_token": None,
    "case_counter": 0,
}


@app.on_event("startup")
def startup():
    dataset = generate_dataset()
    metrics = STATE["scorer"].fit(dataset)
    STATE["ledger"].append({"type": "SYSTEM_START", "detail": "Risk scorer + peer model trained", "metrics": metrics})
    print("ZeroStand trained:", metrics)


@app.get("/api/metrics")
def get_metrics():
    return STATE["scorer"].metrics


@app.get("/api/scenarios")
def scenarios():
    return {"scenarios": SCENARIOS}


@app.get("/api/traces")
def get_traces():
    return STATE["traces"][-50:]


@app.get("/api/tokens")
def get_tokens():
    return STATE["broker"].list_tokens()


def _next_case(scenario):
    cid = STATE["case_counter"]
    STATE["case_counter"] += 1
    base = datetime.now()
    if scenario == "normal":
        return _normal_case(cid, random.choice(NORMAL_USERS), base)
    if scenario == "excessive_scope_request":
        return _excessive_scope_case(cid, random.choice(USERS), base)
    if scenario == "peer_drift_request":
        return _peer_drift_case(cid, random.choice(DRIFT_USERS), base)
    if scenario == "conformance_violation":
        return _conformance_violation_case(cid, random.choice(NORMAL_USERS), base)
    return None


def _run_pipeline(case):
    scorer_result = STATE["scorer"].score(case)
    req_decision = decide_request(scorer_result["risk_score"], case["scope_overreach"])
    STATE["ledger"].append({"type": "REQUEST_DECISION", "case_id": case["case_id"], "user_id": case["user_id"],
                             "action": req_decision["action"], "risk_score": scorer_result["risk_score"]})

    peer_centroid = STATE["scorer"].peer_model.role_centroids.get(case["role"])
    trace = {
        "case_id": case["case_id"], "user_id": case["user_id"], "role": case["role"],
        "action_type": case["action_type"], "target": case["target"], "criticality": case["criticality"],
        "case_type": case["case_type"], "ground_truth": case["request_label"],
        "vectors": {"current": case["current_vector"], "own_recent": case["own_recent_vector"], "peer_centroid": peer_centroid},
        "request": {"risk_score": scorer_result["risk_score"], "top_reasons": scorer_result["top_reasons"],
                     "decision": req_decision["action"], "reason": req_decision["reason"]},
        "token": None, "execution": None,
    }

    if req_decision["action"] != "APPROVE":
        STATE["traces"].append(trace)
        return trace

    token = STATE["broker"].issue(case["user_id"], case["action_type"], case["target"], ttl_seconds=90)
    STATE["last_token"] = token
    STATE["ledger"].append({"type": "TOKEN_ISSUED", "jti": token["jti"], "user_id": case["user_id"],
                             "action_type": case["action_type"]})
    trace["token"] = {"jti": token["jti"], "ttl_seconds": 90}

    redeem_result = STATE["broker"].redeem(token)
    if not redeem_result["valid"]:
        trace["execution"] = {"decision": "DENY", "reason": redeem_result["reason"]}
        STATE["ledger"].append({"type": "TOKEN_REDEEM_FAILED", "jti": token["jti"], "reason": redeem_result["reason"]})
        STATE["traces"].append(trace)
        return trace

    conformance = check_conformance(case["execution_commands"], case["action_type"])
    exec_decision = decide_execution(conformance)
    if exec_decision["action"] == "REVOKE":
        STATE["broker"].revoke(token["jti"])
    STATE["ledger"].append({"type": "EXECUTION_DECISION", "jti": token["jti"], "action": exec_decision["action"],
                             "reason": exec_decision["reason"]})
    trace["execution"] = {"decision": exec_decision["action"], "reason": exec_decision["reason"],
                           "commands": case["execution_commands"], "conformance": conformance}
    STATE["traces"].append(trace)
    return trace


def _run_token_replay():
    token = STATE.get("last_token")
    if not token:
        _run_pipeline(_next_case("normal"))
        token = STATE.get("last_token")
    result = STATE["broker"].redeem(token)
    STATE["ledger"].append({"type": "TOKEN_REPLAY_ATTEMPT", "jti": token["jti"], "valid": result["valid"],
                             "reason": result.get("reason")})
    return {"case_type": "token_replay", "jti": token["jti"], "valid": result["valid"],
            "reason": result.get("reason", "valid -- should not happen")}


@app.post("/api/demo/run")
def run_demo(scenario: str = "normal"):
    if scenario == "token_replay":
        return _run_token_replay()
    case = _next_case(scenario)
    if case is None:
        return {"error": f"unknown scenario, choose from: {SCENARIOS}"}
    return _run_pipeline(case)


@app.get("/api/audit")
def get_audit():
    return STATE["ledger"].chain


@app.get("/api/audit/verify")
def verify_audit():
    return STATE["ledger"].verify_chain()


@app.post("/api/audit/tamper/{index}")
def tamper_audit(index: int):
    STATE["ledger"].tamper_demo(index)
    return {"tampered_index": index}


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent / "static"), html=True), name="static")
