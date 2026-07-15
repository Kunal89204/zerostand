"""
Tamper-evident, quantum-safe decision ledger. ZeroStand has no standing
secrets to protect (that's the point) so the crypto workload shifts here:
protecting the *decision trail* -- every request, approval, denial, token
issuance, and revocation -- which is exactly the long-lived forensic/
regulatory record that harvest-now-decrypt-later targets and that a
classical signature could be retroactively forged against.
"""
import hashlib
import json
import time

from pqcrypto.sign.ml_dsa_65 import generate_keypair as dsa_keypair, sign as dsa_sign, verify as dsa_verify


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, default=str).encode()


class AuditLedger:
    GENESIS_HASH = "0" * 64

    def __init__(self):
        self.pk, self.sk = dsa_keypair()
        self.chain = []

    def append(self, event: dict) -> dict:
        prev_hash = self.chain[-1]["record_hash"] if self.chain else self.GENESIS_HASH
        body = {"seq": len(self.chain), "ts": time.time(), "event": event, "prev_hash": prev_hash}
        record_hash = hashlib.sha3_256(_canonical(body)).hexdigest()
        signature = dsa_sign(self.sk, record_hash.encode())
        entry = {**body, "record_hash": record_hash, "signature": signature.hex()}
        self.chain.append(entry)
        return entry

    def verify_chain(self):
        prev_hash = self.GENESIS_HASH
        for i, entry in enumerate(self.chain):
            body = {k: entry[k] for k in ("seq", "ts", "event", "prev_hash")}
            expected_hash = hashlib.sha3_256(_canonical(body)).hexdigest()
            if expected_hash != entry["record_hash"]:
                return {"valid": False, "broken_at": i, "reason": "hash mismatch (record altered)"}
            if entry["prev_hash"] != prev_hash:
                return {"valid": False, "broken_at": i, "reason": "chain link broken (record inserted/deleted/reordered)"}
            if not dsa_verify(self.pk, entry["record_hash"].encode(), bytes.fromhex(entry["signature"])):
                return {"valid": False, "broken_at": i, "reason": "ML-DSA-65 signature invalid (forged or corrupted)"}
            prev_hash = entry["record_hash"]
        return {"valid": True, "length": len(self.chain), "algorithm": "SHA3-256 hash chain + ML-DSA-65 signatures"}

    def tamper_demo(self, index: int):
        if 0 <= index < len(self.chain):
            self.chain[index]["event"]["tampered"] = True
