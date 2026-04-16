"""Tests for receipt_signer.py"""

import json
import os
import tempfile

from openpilot.tools.lib.receipt_signer import Receipt, ReceiptChain, _jcs_canonicalize


def test_jcs_canonical_sorted_keys():
    assert _jcs_canonicalize({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_jcs_canonical_nested():
    obj = {"z": {"b": 1, "a": 2}, "a": 3}
    result = _jcs_canonicalize(obj)
    assert result == '{"a":3,"z":{"a":2,"b":1}}'


def test_receipt_sign_and_verify():
    receipt = Receipt(payload={"tool": "controlsState", "decision": "allow"})
    receipt.sign("test_key_" + "0" * 56)
    assert receipt.verify("test_key_" + "0" * 56)
    assert receipt.receipt_id.startswith("sha256:")
    assert receipt.signature["alg"] == "HS256"


def test_receipt_tamper_detection():
    receipt = Receipt(payload={"decision": "allow", "speed": 25.0})
    key = "test_key_" + "0" * 56
    receipt.sign(key)
    assert receipt.verify(key)
    receipt.payload["decision"] = "deny"
    assert not receipt.verify(key)


def test_chain_basic():
    chain = ReceiptChain(device_id="test-device", route_id="test-route")
    chain.sign_event("controlsState", {"enabled": True, "vEgo": 25.3})
    chain.sign_event("lateralPlan", {"dPath": [0.0, 0.1, 0.3]})
    chain.sign_event("controlsState", {"enabled": True, "vEgo": 24.8})
    assert chain.length == 3
    assert chain.verify_all()


def test_chain_hash_linking():
    chain = ReceiptChain(device_id="test-device")
    chain.sign_event("a", {"x": 1})
    chain.sign_event("b", {"x": 2})
    receipts = chain.get_receipts()
    assert receipts[0].payload["previousReceiptHash"] is None
    assert receipts[1].payload["previousReceiptHash"] is not None
    assert receipts[1].payload["previousReceiptHash"].startswith("sha256:")


def test_chain_tamper_breaks_verification():
    chain = ReceiptChain(device_id="test-device")
    chain.sign_event("a", {"x": 1})
    chain.sign_event("b", {"x": 2})
    chain.sign_event("c", {"x": 3})
    assert chain.verify_all()
    # Tamper with the middle receipt
    chain._receipts[1].payload["event_type"] = "TAMPERED"
    assert not chain.verify_all()


def test_chain_export_jsonl():
    chain = ReceiptChain(device_id="test-device")
    chain.sign_event("controlsState", {"vEgo": 25.0})
    chain.sign_event("controlsState", {"vEgo": 24.5})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        chain.export_jsonl(path)
        with open(path) as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "payload" in parsed
            assert "signature" in parsed
            assert parsed["payload"]["type"] == "openpilot:decision"
    finally:
        os.unlink(path)


def test_chain_sequence_numbering():
    chain = ReceiptChain(device_id="test-device")
    for i in range(5):
        chain.sign_event("e", {"i": i})
    receipts = chain.get_receipts()
    for i, r in enumerate(receipts):
        assert r.payload["sequence"] == i + 1


def test_receipt_payload_hashes_event_data():
    chain = ReceiptChain(device_id="test-device")
    r = chain.sign_event("controlsState", {"steeringAngleDeg": -2.1})
    assert "event_hash" in r.payload
    assert r.payload["event_hash"].startswith("sha256:")
    # Event hash should NOT contain raw sensor data
    assert "steeringAngleDeg" not in json.dumps(r.payload)


if __name__ == "__main__":
    test_jcs_canonical_sorted_keys()
    test_jcs_canonical_nested()
    test_receipt_sign_and_verify()
    test_receipt_tamper_detection()
    test_chain_basic()
    test_chain_hash_linking()
    test_chain_tamper_breaks_verification()
    test_chain_export_jsonl()
    test_chain_sequence_numbering()
    test_receipt_payload_hashes_event_data()
    print("All 10 tests passed.")
