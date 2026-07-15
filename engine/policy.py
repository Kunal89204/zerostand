def decide_request(risk_score: float, scope_overreach: bool):
    if scope_overreach:
        return {"action": "DENY", "stage": "request",
                "reason": "Requested action falls outside this role's entitlement catalog -- denied before any token is issued."}
    if risk_score >= 0.70:
        return {"action": "DENY", "stage": "request",
                "reason": f"Risk score {risk_score} exceeds deny threshold."}
    if risk_score >= 0.40:
        return {"action": "STEP_UP", "stage": "request",
                "reason": f"Risk score {risk_score} requires manager step-up approval before a token is issued."}
    return {"action": "APPROVE", "stage": "request", "reason": "Within normal risk range -- token issued."}


def decide_execution(conformance_result: dict):
    if conformance_result["violation"]:
        buckets = sorted({f["bucket"] for f in conformance_result["flagged_commands"]})
        return {"action": "REVOKE", "stage": "execution",
                "reason": f"Executed command(s) outside declared scope (expected {conformance_result['expected_buckets']}, "
                          f"saw {buckets}) -- token revoked immediately."}
    return {"action": "ALLOW", "stage": "execution", "reason": "All executed commands within declared scope."}
