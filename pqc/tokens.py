"""
Ephemeral capability tokens -- the core artefact of "zero standing
privilege." Nothing long-lived is ever handed out: a token is scoped to one
requestor, one action, one target, expires in seconds, and can be redeemed
exactly once. Payload is ML-DSA-65 signed (so it can't be forged) and the
whole envelope is encrypted with the same ML-KEM-768 + X25519 hybrid scheme
as a credential vault would use (so it can't be read in transit either) --
except there's no standing secret inside it to steal; only a narrow,
short-lived permission slip.

Replay and expiry are enforced here, at the cryptographic/protocol layer,
independent of any ML model -- so even a bypassed or fooled risk model still
can't get a stale or reused token honored.
"""
import json
import os
import time
import uuid

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from pqcrypto.kem.ml_kem_768 import generate_keypair as kem_keypair, encrypt as kem_encapsulate, decrypt as kem_decapsulate
from pqcrypto.sign.ml_dsa_65 import generate_keypair as dsa_keypair, sign as dsa_sign, verify as dsa_verify


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode()


def _derive_key(kem_ss: bytes, x_ss: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA3_256(), length=32, salt=None, info=info).derive(kem_ss + x_ss)


class TokenBroker:
    def __init__(self):
        self.kem_pk, self.kem_sk = kem_keypair()
        self.x25519_sk = X25519PrivateKey.generate()
        self.x25519_pk = self.x25519_sk.public_key()
        self.dsa_pk, self.dsa_sk = dsa_keypair()
        self.consumed_jti = set()
        self.issued = {}

    def issue(self, requestor: str, action_type: str, target: str, ttl_seconds: int = 90):
        jti = uuid.uuid4().hex
        now = time.time()
        payload = {"jti": jti, "requestor": requestor, "action_type": action_type,
                   "target": target, "iat": now, "exp": now + ttl_seconds}
        signature = dsa_sign(self.dsa_sk, _canonical(payload))
        envelope = {"payload": payload, "signature": signature.hex()}
        envelope_bytes = _canonical(envelope)

        kem_ct, kem_ss = kem_encapsulate(self.kem_pk)
        eph_sk = X25519PrivateKey.generate()
        eph_pub_bytes = eph_sk.public_key().public_bytes_raw()
        x_ss = eph_sk.exchange(self.x25519_pk)
        key = _derive_key(kem_ss, x_ss, info=jti.encode())
        nonce = os.urandom(12)
        ciphertext = AESGCM(key).encrypt(nonce, envelope_bytes, associated_data=jti.encode())

        token = {"jti": jti, "kem_ct": kem_ct, "eph_pub": eph_pub_bytes, "nonce": nonce, "ciphertext": ciphertext}
        self.issued[jti] = {"requestor": requestor, "action_type": action_type, "target": target,
                             "issued_at": now, "exp": now + ttl_seconds, "status": "active"}
        return token

    def redeem(self, token: dict):
        jti = token["jti"]
        try:
            kem_ss = kem_decapsulate(self.kem_sk, token["kem_ct"])
            x_ss = self.x25519_sk.exchange(X25519PublicKey.from_public_bytes(token["eph_pub"]))
            key = _derive_key(kem_ss, x_ss, info=jti.encode())
            envelope_bytes = AESGCM(key).decrypt(token["nonce"], token["ciphertext"], associated_data=jti.encode())
            envelope = json.loads(envelope_bytes)
        except Exception:
            return {"valid": False, "reason": "decryption/authentication failed -- token corrupted or forged"}

        payload, signature = envelope["payload"], bytes.fromhex(envelope["signature"])
        if not dsa_verify(self.dsa_pk, _canonical(payload), signature):
            return {"valid": False, "reason": "ML-DSA-65 signature invalid -- token forged or tampered"}
        if jti in self.consumed_jti:
            return {"valid": False, "reason": "replay detected -- this token's single-use nonce was already consumed"}
        if time.time() > payload["exp"]:
            return {"valid": False, "reason": "token expired"}

        self.consumed_jti.add(jti)
        if jti in self.issued:
            self.issued[jti]["status"] = "consumed"
        return {"valid": True, "payload": payload}

    def revoke(self, jti: str):
        self.consumed_jti.add(jti)
        if jti in self.issued:
            self.issued[jti]["status"] = "revoked"

    def list_tokens(self):
        now = time.time()
        out = []
        for jti, meta in self.issued.items():
            status = meta["status"]
            if status == "active" and now > meta["exp"]:
                status = "expired"
            out.append({"jti": jti, **meta, "status": status, "seconds_remaining": max(0, round(meta["exp"] - now, 1))})
        return out
