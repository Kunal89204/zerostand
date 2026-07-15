"""
Execution-time conformance check: deterministic, not ML. A request declared
"restart_service" grants only what that action template expects
(config + read_only); anything executed under the token outside that scope
-- and specifically in a high-risk bucket -- is a conformance violation and
revokes the token immediately.

Deliberately rule-based rather than statistical: this is the layer where
you want a provably-correct answer, not a probability. Pairing a
deterministic scope check with the probabilistic request-stage risk model
is the "defense in depth, not one model doing everything" pitch.
"""
from data.simulator import ACTION_TEMPLATES, COMMAND_POOL, HIGH_RISK_BUCKETS, ALWAYS_ALLOWED

CMD_TO_BUCKET = {cmd: bucket for bucket, cmds in COMMAND_POOL.items() for cmd in cmds}


def bucket_of(cmd: str) -> str:
    return CMD_TO_BUCKET.get(cmd, "read_only")


def check_conformance(commands, action_type):
    template = ACTION_TEMPLATES[action_type]
    expected = set(template["expected_buckets"]) | ALWAYS_ALLOWED
    flagged = []
    for c in commands:
        b = bucket_of(c)
        if b not in expected and b in HIGH_RISK_BUCKETS:
            flagged.append({"command": c, "bucket": b})
    return {
        "violation": len(flagged) > 0,
        "flagged_commands": flagged,
        "expected_buckets": sorted(expected),
        "commands_checked": len(commands),
    }
