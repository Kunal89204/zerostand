"""
Synthetic data for ZeroStand's just-in-time access broker.

Unlike a continuous session-monitoring dataset, each "case" here is a full
request lifecycle: a declared task (role, action type, target), the
requestor's behavioral profile at request time, and (for some scenarios)
the commands actually executed once a token was issued. Four scenario
families, each defeating a different layer of the pipeline:

  normal                  -- request matches role/history, execution matches
                             the declared action. Nothing interesting happens
                             (which is the point -- most traffic should be boring).
  excessive_scope_request -- the *request itself* asks for an action outside
                             the requestor's role/history. Caught before any
                             token is ever issued.
  peer_drift_request      -- requestor's own recent behavior is internally
                             consistent (self-deviation low) but has drifted
                             away from their peer group's baseline
                             (peer-deviation high). A model that only
                             compares you to your own history would miss this.
  conformance_violation   -- the request looks completely normal and is
                             approved, but the commands actually executed
                             under the token fall outside the declared
                             action's expected scope. Caught at execution
                             time, not request time.

token_replay is not a labeled dataset case -- it's a live protocol-level
demo action performed directly against an already-issued token.
"""
import random
from datetime import datetime, timedelta

random.seed(23)

BUCKET_NAMES = ["read_only", "config", "data_move", "priv_esc", "destructive", "log_tamper", "vault_access"]
HIGH_RISK_BUCKETS = {"data_move", "priv_esc", "destructive", "log_tamper", "vault_access"}
ALWAYS_ALLOWED = {"read_only"}

COMMAND_POOL = {
    "read_only":    ["ls -la", "cat /var/log/app.log", "select * from accounts limit 10", "show tables", "systemctl status api"],
    "config":       ["systemctl restart nginx", "vim /etc/app/config.yml", "chmod 644 config.yml"],
    "data_move":    ["mysqldump prod > dump.sql", "scp dump.sql ext@vendor:/tmp/", "aws s3 cp dump.sql s3://ext-bucket/"],
    "priv_esc":     ["sudo su -", "grant all privileges on *.* to 'tmp'@'%'", "usermod -aG sudo svc_x"],
    "destructive":  ["drop table audit_log", "rm -rf /var/backups/*", "truncate table transactions"],
    "log_tamper":   ["systemctl stop auditd", "history -c", "auditctl -e 0"],
    "vault_access": ["vault read secret/prod/db-creds", "vault read secret/prod/api-keys"],
}

ROLES = ["DBA", "SysAdmin", "Vendor"]

ROLE_CENTROID = {
    "DBA":      {"read_only": 0.60, "config": 0.28, "data_move": 0.06, "priv_esc": 0.02, "destructive": 0.01, "log_tamper": 0.01, "vault_access": 0.02},
    "SysAdmin": {"read_only": 0.50, "config": 0.42, "data_move": 0.03, "priv_esc": 0.02, "destructive": 0.01, "log_tamper": 0.01, "vault_access": 0.01},
    "Vendor":   {"read_only": 0.82, "config": 0.10, "data_move": 0.04, "priv_esc": 0.00, "destructive": 0.00, "log_tamper": 0.00, "vault_access": 0.04},
}

EXFIL_LIKE_VECTOR = {"read_only": 0.20, "config": 0.05, "data_move": 0.45, "priv_esc": 0.05,
                      "destructive": 0.00, "log_tamper": 0.00, "vault_access": 0.25}

ACTION_TEMPLATES = {
    "restart_service":       {"expected_buckets": ["config", "read_only"], "criticality": "medium", "typical_roles": ["SysAdmin", "DBA"], "target": "web-prod-01"},
    "run_migration":         {"expected_buckets": ["read_only", "config", "data_move"], "criticality": "high", "typical_roles": ["DBA"], "target": "db-prod-01"},
    "incident_investigation": {"expected_buckets": ["read_only"], "criticality": "medium", "typical_roles": ["SysAdmin", "DBA"], "target": "web-prod-02"},
    "vendor_report_pull":    {"expected_buckets": ["read_only"], "criticality": "low", "typical_roles": ["Vendor"], "target": "reporting-01"},
}

N_PER_ROLE = 10
USERS = []
DRIFT_USER_IDS = set()
for role in ROLES:
    for i in range(N_PER_ROLE):
        uid = f"{role.lower()}_{i}"
        USERS.append({"id": uid, "role": role})
    drift_id = f"{role.lower()}_0"  # first user of each role is the slow-drifter
    DRIFT_USER_IDS.add(drift_id)

USER_BY_ID = {u["id"]: u for u in USERS}
NORMAL_USERS = [u for u in USERS if u["id"] not in DRIFT_USER_IDS]
DRIFT_USERS = [u for u in USERS if u["id"] in DRIFT_USER_IDS]


def _sample_vector(centroid, noise=0.05):
    v = {k: max(0.0, random.gauss(val, noise)) for k, val in centroid.items()}
    s = sum(v.values()) or 1.0
    return {k: val / s for k, val in v.items()}


def _interpolate(a, b, t):
    v = {k: a[k] * (1 - t) + b.get(k, 0) * t for k in a}
    s = sum(v.values()) or 1.0
    return {k: val / s for k, val in v.items()}


def _conforming_commands(template, n):
    allowed = list(set(template["expected_buckets"]) | ALWAYS_ALLOWED)
    return [random.choice(COMMAND_POOL[random.choice(allowed)]) for _ in range(n)]


def _violating_commands(template, n_conform=4, n_violate=2):
    conforming = _conforming_commands(template, n_conform)
    off_scope = [b for b in HIGH_RISK_BUCKETS if b not in template["expected_buckets"]]
    violating = [random.choice(COMMAND_POOL[random.choice(off_scope)]) for _ in range(n_violate)]
    combined = conforming + violating
    random.shuffle(combined)
    return combined


def _typical_action_for(role):
    options = [a for a, t in ACTION_TEMPLATES.items() if role in t["typical_roles"]]
    return random.choice(options)


def _case(cid, user, action_type, current_vector, own_recent_vector, scope_overreach,
          request_label, case_type, commands, ts):
    template = ACTION_TEMPLATES[action_type]
    return {
        "case_id": cid, "user_id": user["id"], "role": user["role"], "action_type": action_type,
        "target": template["target"], "criticality": template["criticality"],
        "timestamp": ts.isoformat(),
        "current_vector": current_vector, "own_recent_vector": own_recent_vector,
        "scope_overreach": scope_overreach,
        "request_label": request_label,  # "normal" | "anomalous" -- ground truth for the request-stage model
        "case_type": case_type,
        "execution_commands": commands,
    }


def _normal_case(cid, user, base_time):
    role = user["role"]
    action_type = _typical_action_for(role)
    template = ACTION_TEMPLATES[action_type]
    vector = _sample_vector(ROLE_CENTROID[role])
    own_recent = _sample_vector(ROLE_CENTROID[role])
    commands = _conforming_commands(template, random.randint(5, 8))
    ts = base_time + timedelta(minutes=random.randint(0, 60 * 24 * 30))
    return _case(cid, user, action_type, vector, own_recent, False, "normal", "normal", commands, ts)


def _excessive_scope_case(cid, user, base_time):
    role = user["role"]
    mismatched = [a for a, t in ACTION_TEMPLATES.items() if role not in t["typical_roles"]]
    action_type = random.choice(mismatched)
    template = ACTION_TEMPLATES[action_type]
    vector = _sample_vector(ROLE_CENTROID[role])
    own_recent = _sample_vector(ROLE_CENTROID[role])
    commands = _conforming_commands(template, 6)  # hypothetical -- request should be denied before this ever runs
    ts = base_time + timedelta(minutes=random.randint(0, 60 * 24 * 30))
    return _case(cid, user, action_type, vector, own_recent, True, "anomalous", "excessive_scope_request", commands, ts)


def _peer_drift_case(cid, user, base_time):
    role = user["role"]
    action_type = _typical_action_for(role)
    progress = random.uniform(0.55, 0.9)
    current_vector = _interpolate(ROLE_CENTROID[role], EXFIL_LIKE_VECTOR, progress)
    own_recent_vector = _interpolate(ROLE_CENTROID[role], EXFIL_LIKE_VECTOR, max(0.0, progress - 0.15))
    template = ACTION_TEMPLATES[action_type]
    commands = _conforming_commands(template, 6)
    ts = base_time + timedelta(minutes=random.randint(0, 60 * 24 * 30))
    return _case(cid, user, action_type, current_vector, own_recent_vector, False, "anomalous", "peer_drift_request", commands, ts)


def _conformance_violation_case(cid, user, base_time):
    role = user["role"]
    action_type = _typical_action_for(role)
    template = ACTION_TEMPLATES[action_type]
    vector = _sample_vector(ROLE_CENTROID[role])
    own_recent = _sample_vector(ROLE_CENTROID[role])
    commands = _violating_commands(template)
    ts = base_time + timedelta(minutes=random.randint(0, 60 * 24 * 30))
    return _case(cid, user, action_type, vector, own_recent, False, "normal", "conformance_violation", commands, ts)


def generate_dataset(n_normal=300, n_excessive=40, n_drift=30, n_conformance=40, start_date="2026-06-01"):
    base = datetime.fromisoformat(start_date)
    cases, cid = [], 0
    for _ in range(n_normal):
        cases.append(_normal_case(cid, random.choice(NORMAL_USERS), base)); cid += 1
    for _ in range(n_excessive):
        cases.append(_excessive_scope_case(cid, random.choice(USERS), base)); cid += 1
    for _ in range(n_drift):
        cases.append(_peer_drift_case(cid, random.choice(DRIFT_USERS), base)); cid += 1
    for _ in range(n_conformance):
        cases.append(_conformance_violation_case(cid, random.choice(NORMAL_USERS), base)); cid += 1
    random.shuffle(cases)
    return cases


if __name__ == "__main__":
    import json
    data = generate_dataset()
    with open("data/cases.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} cases -> data/cases.json")
