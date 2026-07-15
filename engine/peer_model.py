"""
Peer-group behavioral model. Deliberately separate from self-baselining: a
model that only ever compares a user to their own history will happily
absorb a slow drift, since each new session only differs slightly from the
last one it's already accepted as normal. Comparing the same session against
a peer-group centroid (built from OTHER members of the same role) catches
that drift the moment it puts the user meaningfully out of step with peers,
even though nothing about the change looks sudden to their own history.
"""
from data.simulator import BUCKET_NAMES


def l1_distance(vec_a: dict, vec_b: dict) -> float:
    return sum(abs(vec_a.get(k, 0) - vec_b.get(k, 0)) for k in BUCKET_NAMES)


class PeerModel:
    def __init__(self):
        self.role_centroids = {}

    def fit(self, train_cases):
        """Centroids built only from request-stage-*normal*, held-out-safe cases -- a
        drifting or overreaching case must never contaminate the peer baseline it's
        being judged against."""
        by_role = {}
        for c in train_cases:
            if c["request_label"] == "normal":
                by_role.setdefault(c["role"], []).append(c["current_vector"])
        for role, vectors in by_role.items():
            centroid = {k: sum(v[k] for v in vectors) / len(vectors) for k in BUCKET_NAMES}
            self.role_centroids[role] = centroid

    def peer_deviation(self, vector, role):
        centroid = self.role_centroids.get(role)
        if not centroid:
            return 0.0
        return l1_distance(vector, centroid)

    def self_deviation(self, vector, own_recent_vector):
        return l1_distance(vector, own_recent_vector)
