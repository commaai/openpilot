import os
import tempfile

from openpilot.system.updated.common import get_consistent_flag, set_consistent_flag


def test_flag_initially_false():
    with tempfile.TemporaryDirectory() as tmp:
        assert get_consistent_flag(tmp) is False


def test_flag_set_true_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        set_consistent_flag(tmp, True)
        assert get_consistent_flag(tmp) is True
        assert os.path.exists(os.path.join(tmp, ".overlay_consistent"))


def test_flag_set_false_removes_file():
    with tempfile.TemporaryDirectory() as tmp:
        set_consistent_flag(tmp, True)
        set_consistent_flag(tmp, False)
        assert get_consistent_flag(tmp) is False
        assert not os.path.exists(os.path.join(tmp, ".overlay_consistent"))


def test_flag_set_false_when_missing_is_safe():
    with tempfile.TemporaryDirectory() as tmp:
        set_consistent_flag(tmp, False)
        assert get_consistent_flag(tmp) is False
