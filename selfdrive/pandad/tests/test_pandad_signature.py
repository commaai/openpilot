import pytest


from openpilot.selfdrive.pandad.pandad import get_expected_signature


class DummyPanda:
    @staticmethod
    def get_signature_from_firmware(fn):
        return b"abc123"


def test_get_expected_signature_success(monkeypatch):
    monkeypatch.setattr(
        "openpilot.selfdrive.pandad.pandad.Panda",
        DummyPanda,
    )

    result = get_expected_signature()
    assert result == b"abc123"


class FailingPanda:
    @staticmethod
    def get_signature_from_firmware(fn):
        raise Exception("failure")


def test_get_expected_signature_failure(monkeypatch):
    monkeypatch.setattr(
        "openpilot.selfdrive.pandad.pandad.Panda",
        FailingPanda,
    )

    result = get_expected_signature()
    assert result == b""

def test_get_expected_signature_returns_bytes(monkeypatch):
    class DummyPanda:
        @staticmethod
        def get_signature_from_firmware(fn):
            return b"123"

    monkeypatch.setattr(
        "openpilot.selfdrive.pandad.pandad.Panda",
        DummyPanda,
    )

    result = get_expected_signature()
    assert isinstance(result, bytes)
