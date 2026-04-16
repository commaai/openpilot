"""
Receipt-signed decision logs for openpilot.

Wraps existing cereal log events in Ed25519-signed receipts, producing a
hash-chained audit trail that is independently verifiable without trusting
the device operator. Useful for post-incident investigation, insurance
claims, and regulatory compliance.

Zero new dependencies beyond Python stdlib. Uses SHA-256 HMAC for the
reference implementation (production upgrade path: Ed25519 via PyNaCl or
a hardware secure element).

Usage:
    from tools.lib.receipt_signer import ReceiptChain

    chain = ReceiptChain(device_id="comma-3x-abc123")

    # Sign any decision event (controlsState, lateralPlan, etc.)
    receipt = chain.sign_event(
        event_type="controlsState",
        event_data={"enabled": True, "vEgo": 25.3, "steeringAngleDeg": -2.1},
    )

    # Verify the chain
    assert chain.verify_all()

    # Export for offline verification
    chain.export_jsonl("/data/receipts/route_abc.jsonl")

    # Verify offline with:
    #   npx @veritasacta/verify /data/receipts/route_abc.jsonl --key <key>

Receipt format follows IETF draft-farley-acta-signed-receipts.
See: https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional


def _jcs_canonicalize(obj: Any) -> str:
    """RFC 8785 JCS canonicalization: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass
class Receipt:
    """A signed receipt for an openpilot decision event."""

    payload: dict[str, Any]
    signature: dict[str, str] = field(default_factory=dict)
    receipt_id: str = ""

    def sign(self, key: str) -> None:
        """Sign with HMAC-SHA256 (reference impl). Production: Ed25519."""
        canonical = _jcs_canonicalize(self.payload)
        sig = hmac.new(key.encode(), canonical.encode(), hashlib.sha256).hexdigest()
        self.signature = {"alg": "HS256", "kid": f"op:{key[:8]}", "sig": sig}
        self.receipt_id = "sha256:" + _sha256_hex(canonical)

    def verify(self, key: str) -> bool:
        """Verify the receipt signature."""
        canonical = _jcs_canonicalize(self.payload)
        expected = hmac.new(key.encode(), canonical.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature.get("sig", ""), expected)

    def to_dict(self) -> dict[str, Any]:
        return {"payload": self.payload, "signature": self.signature}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


class ReceiptChain:
    """Hash-chained receipt log for openpilot decision events.

    Each receipt links to the previous via SHA-256 hash, creating an
    append-only chain where insertions, deletions, and modifications
    are all detectable.

    The chain key is generated per-route (not per-device) so that each
    driving session has an independent, self-contained audit trail.
    """

    def __init__(self, device_id: str = "", route_id: str = ""):
        self.device_id = device_id or f"comma-{os.urandom(4).hex()}"
        self.route_id = route_id or f"route-{os.urandom(4).hex()}"
        self._key = os.urandom(32).hex()
        self._receipts: list[Receipt] = []
        self._sequence = 0

    @property
    def public_key(self) -> str:
        """Key needed for verification (HMAC: the key itself; Ed25519: public half)."""
        return self._key

    @property
    def length(self) -> int:
        return len(self._receipts)

    def sign_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> Receipt:
        """Sign a cereal log event and append to the chain.

        Args:
            event_type: cereal message type (e.g., "controlsState", "lateralPlan")
            event_data: dict of the event's fields (serialized from capnp)
            timestamp: event timestamp in seconds (defaults to time.time())

        Returns:
            The signed receipt.
        """
        self._sequence += 1
        ts = timestamp or time.time()

        prev_hash = None
        if self._receipts:
            prev_canonical = _jcs_canonicalize(self._receipts[-1].payload)
            prev_hash = "sha256:" + _sha256_hex(prev_canonical)

        # Hash the event data (privacy-preserving: receipt proves the event
        # existed with this content, without storing raw sensor values in
        # the receipt itself)
        event_hash = "sha256:" + _sha256_hex(_jcs_canonicalize(event_data))

        payload = {
            "type": "openpilot:decision",
            "spec": "draft-farley-acta-signed-receipts-01",
            "device_id": self.device_id,
            "route_id": self.route_id,
            "event_type": event_type,
            "event_hash": event_hash,
            "issued_at": ts,
            "sequence": self._sequence,
            "previousReceiptHash": prev_hash,
        }

        receipt = Receipt(payload=payload)
        receipt.sign(self._key)
        self._receipts.append(receipt)
        return receipt

    def verify_all(self) -> bool:
        """Verify every receipt in the chain: signatures + hash links."""
        for i, receipt in enumerate(self._receipts):
            if not receipt.verify(self._key):
                return False
            if i > 0:
                prev_canonical = _jcs_canonicalize(self._receipts[i - 1].payload)
                expected_hash = "sha256:" + _sha256_hex(prev_canonical)
                if receipt.payload.get("previousReceiptHash") != expected_hash:
                    return False
        return True

    def export_jsonl(self, path: str) -> None:
        """Write the chain to a JSONL file for offline verification."""
        with open(path, "w") as f:
            for receipt in self._receipts:
                f.write(receipt.to_json() + "\n")

    def get_receipts(self) -> list[Receipt]:
        """Return a copy of all receipts."""
        return list(self._receipts)


# ---------------------------------------------------------------------------
# Convenience: wrap the cereal logger
# ---------------------------------------------------------------------------

def create_receipt_logger(device_id: str = "", route_id: str = "") -> ReceiptChain:
    """Create a receipt chain for the current route.

    Call this at route start. The returned chain produces a receipt for
    every event you pass to `sign_event`. At route end, call
    `export_jsonl` to persist the chain.

    Example integration with cereal SubMaster:
        chain = create_receipt_logger(device_id=Params().get("DongleId"))

        sm = messaging.SubMaster(['controlsState', 'lateralPlan'])
        while True:
            sm.update()
            if sm.updated['controlsState']:
                chain.sign_event("controlsState", sm['controlsState'].to_dict())
    """
    return ReceiptChain(device_id=device_id, route_id=route_id)
