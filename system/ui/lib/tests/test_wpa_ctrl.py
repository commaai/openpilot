"""Tests for wpa_ctrl parsing helpers and constants."""
import threading
import time

import pytest

from openpilot.system.ui.lib.wpa_ctrl import (
  RECV_BUF_SIZE,
  SecurityType,
  WpaCtrl,
  decode_ssid,
  parse_scan_results,
  parse_status,
  flags_to_security_type,
  dbm_to_percent,
)


class TestParseStatus:
  def test_basic(self):
    raw = "wpa_state=COMPLETED\nssid=MyNet\nip_address=10.0.0.5\n"
    d = parse_status(raw)
    assert d["wpa_state"] == "COMPLETED"
    assert d["ssid"] == "MyNet"
    assert d["ip_address"] == "10.0.0.5"

  def test_value_with_equals(self):
    raw = "ssid=My=Network\n"
    d = parse_status(raw)
    assert d["ssid"] == "My=Network"

  def test_empty(self):
    assert parse_status("") == {}

  def test_ssid_decoded(self):
    # STATUS emits ssid via wpa_ssid_txt (printf_encode), so non-ASCII
    # bytes come through escaped and must be decoded as UTF-8.
    raw = "wpa_state=COMPLETED\nssid=caf\\xc3\\xa9\nip_address=10.0.0.5\n"
    d = parse_status(raw)
    assert d["ssid"] == "café"
    assert d["ssid"].encode("utf-8") == b"caf\xc3\xa9"

  def test_ssid_with_embedded_quote(self):
    raw = 'ssid=My \\"Home\\"\n'
    assert parse_status(raw)["ssid"] == 'My "Home"'

  def test_non_ssid_keys_untouched(self):
    # Only the ssid value is decoded; other fields must pass through.
    raw = "bssid=00:11:22:33:44:55\nssid=\\x41\n"
    d = parse_status(raw)
    assert d["bssid"] == "00:11:22:33:44:55"
    assert d["ssid"] == "A"


class TestFlagsToSecurityType:
  @pytest.mark.parametrize("flags,expected", [
    ("[WPA2-PSK-CCMP][ESS]", SecurityType.WPA),
    ("[RSN-PSK-CCMP]", SecurityType.WPA),
    ("[WPA-PSK-TKIP]", SecurityType.WPA),
    # WPA2/WPA3 transitional: PSK takes precedence so the network is connectable.
    ("[WPA2-PSK-CCMP][SAE]", SecurityType.WPA),
    ("[RSN-PSK-CCMP][SAE-CCMP]", SecurityType.WPA),
    # WPA3-only (SAE) isn't connectable on the current kernel; surface as unsupported
    # rather than prompting for a password that will fail the handshake.
    ("[SAE]", SecurityType.UNSUPPORTED),
    ("[SAE-CCMP]", SecurityType.UNSUPPORTED),
    ("[ESS]", SecurityType.OPEN),
    ("", SecurityType.OPEN),
    ("[WPA2-EAP-CCMP]", SecurityType.UNSUPPORTED),
    ("[802.1X]", SecurityType.UNSUPPORTED),
  ])
  def test_security_types(self, flags, expected):
    assert flags_to_security_type(flags) == expected


class TestDbmToPercent:
  def test_boundaries(self):
    assert dbm_to_percent(-100) == 0
    assert dbm_to_percent(-40) == 100

  def test_clamps(self):
    assert dbm_to_percent(-120) == 0
    assert dbm_to_percent(-30) == 100

  def test_mid(self):
    assert dbm_to_percent(-70) == 50

  def test_nm_parity(self):
    # matches nm_wifi_utils_level_to_quality test vectors
    assert dbm_to_percent(-74) == 44
    assert dbm_to_percent(-81) == 32
    assert dbm_to_percent(-92) == 14


class TestParseScanResults:
  HEADER = "bssid / frequency / signal level / flags / ssid\n"

  def test_basic(self):
    raw = self.HEADER + "00:11:22:33:44:55\t2437\t-65\t[WPA2-PSK-CCMP][ESS]\tMyNetwork\n"
    results = parse_scan_results(raw)
    assert len(results) == 1
    r = results[0]
    assert r.bssid == "00:11:22:33:44:55"
    assert r.freq == 2437
    assert r.signal == -65
    assert r.ssid == "MyNetwork"

  def test_hidden_ssid(self):
    raw = self.HEADER + "00:11:22:33:44:55\t2437\t-65\t[ESS]\t\n"
    results = parse_scan_results(raw)
    assert len(results) == 1
    assert results[0].ssid == ""

  def test_null_padded_hidden_ssid(self):
    """Some APs broadcast 32 null bytes instead of an empty SSID; wpa_supplicant
    emits them as `\\x00` escapes. Those should normalize to empty so the UI
    filters them out the same as empty hidden networks."""
    padded = "\\x00" * 32
    raw = self.HEADER + f"00:11:22:33:44:55\t2437\t-65\t[ESS]\t{padded}\n"
    results = parse_scan_results(raw)
    assert len(results) == 1
    assert results[0].ssid == ""

  def test_escaped_ssid(self):
    # \xc3\xa9 is UTF-8 for é — decode_ssid must reinterpret as UTF-8,
    # not one-codepoint-per-byte (Latin-1), or SET_NETWORK round-trip
    # will re-encode as 7 bytes and fail to match the 5-byte AP SSID.
    raw = self.HEADER + '00:11:22:33:44:55\t2437\t-65\t[ESS]\tcaf\\xc3\\xa9 \\"home\\"\n'
    results = parse_scan_results(raw)
    assert len(results) == 1
    assert results[0].ssid == 'café "home"'
    # Round-trip: the decoded str must re-encode to the original AP bytes.
    assert results[0].ssid.encode("utf-8") == b'caf\xc3\xa9 "home"'

  def test_preserves_trailing_space_in_ssid(self):
    """SSIDs may legally end with spaces and wpa_supplicant leaves printable
    spaces unescaped. Stripping the whole payload would clip the last line's
    trailing-space SSID, so connect/forget would target the wrong name."""
    raw = self.HEADER + "00:11:22:33:44:55\t2437\t-65\t[ESS]\tMyNet \n"
    results = parse_scan_results(raw)
    assert len(results) == 1
    assert results[0].ssid == "MyNet "

  def test_missing_ssid_field(self):
    raw = self.HEADER + "00:11:22:33:44:55\t2437\t-65\t[ESS]\n"
    results = parse_scan_results(raw)
    assert len(results) == 1
    assert results[0].ssid == ""

  def test_malformed_lines_skipped(self):
    raw = self.HEADER + "garbage\n" + "00:11:22:33:44:55\t2437\t-65\t[ESS]\tGood\n"
    results = parse_scan_results(raw)
    assert len(results) == 1
    assert results[0].ssid == "Good"

  def test_large_scan_fits_in_recv_buffer(self):
    """A dense AP environment can return many results. Verify they parse
    correctly and that RECV_BUF_SIZE is large enough for a realistic worst case."""
    lines = [self.HEADER.strip()]
    for i in range(200):
      bssid = f"00:11:22:33:{i // 256:02x}:{i % 256:02x}"
      ssid = f"Network_{i:03d}_with_a_longer_name_padding"
      lines.append(f"{bssid}\t2437\t{-30 - (i % 70)}\t[WPA2-PSK-CCMP][ESS]\t{ssid}")
    raw = "\n".join(lines) + "\n"

    # Ensure the payload fits in our buffer
    assert len(raw.encode()) < RECV_BUF_SIZE, (
      f"200-AP scan result ({len(raw.encode())} bytes) exceeds RECV_BUF_SIZE ({RECV_BUF_SIZE})"
    )

    results = parse_scan_results(raw)
    assert len(results) == 200
    assert results[0].ssid == "Network_000_with_a_longer_name_padding"
    assert results[199].ssid == "Network_199_with_a_longer_name_padding"

  def test_old_buffer_would_truncate(self):
    """Demonstrate that 4096 bytes is insufficient for dense scan results."""
    lines = ["header"]
    for i in range(200):
      bssid = f"00:11:22:33:{i // 256:02x}:{i % 256:02x}"
      ssid = f"Network_{i:03d}_with_a_longer_name_padding"
      lines.append(f"{bssid}\t2437\t{-30 - (i % 70)}\t[WPA2-PSK-CCMP][ESS]\t{ssid}")
    raw = "\n".join(lines) + "\n"
    assert len(raw.encode()) > 4096, "Test assumes 200 APs exceed 4096 bytes"


class TestDecodeSsid:
  """Must match wpa_supplicant printf_decode (hostap/src/utils/common.c:526)."""

  def test_plain(self):
    assert decode_ssid("MyNetwork") == "MyNetwork"

  def test_empty(self):
    assert decode_ssid("") == ""

  def test_hex_two_digit(self):
    assert decode_ssid("\\x41\\x42") == "AB"

  def test_hex_uppercase(self):
    # A lone 0xFF is not valid UTF-8; decodes to U+FFFD replacement.
    # (2-digit uppercase-hex parsing itself is covered by test_utf8_multibyte.)
    assert decode_ssid("\\xFF") == "\ufffd"

  def test_utf8_multibyte(self):
    # "café" (UTF-8: 63 61 66 c3 a9) must decode as 4 codepoints, not 5.
    decoded = decode_ssid("caf\\xc3\\xa9")
    assert decoded == "café"
    assert len(decoded) == 4
    assert decoded.encode("utf-8") == b"caf\xc3\xa9"

  def test_utf8_three_byte(self):
    # "日本" (UTF-8: e6 97 a5 e6 9c ac) — common 3-byte CJK case.
    decoded = decode_ssid("\\xe6\\x97\\xa5\\xe6\\x9c\\xac")
    assert decoded == "日本"
    assert decoded.encode("utf-8") == b"\xe6\x97\xa5\xe6\x9c\xac"

  def test_utf8_emoji(self):
    # 4-byte UTF-8 sequence (🚗 U+1F697 = f0 9f 9a 97).
    decoded = decode_ssid("\\xf0\\x9f\\x9a\\x97")
    assert decoded == "🚗"
    assert decoded.encode("utf-8") == b"\xf0\x9f\x9a\x97"

  def test_invalid_utf8_replaced(self):
    # A lone high byte is invalid UTF-8 → U+FFFD, not Latin-1.
    assert decode_ssid("\\x80") == "\ufffd"

  def test_hex_one_digit_fallback(self):
    # `\x1Z`: hex2byte("1Z") fails (Z not hex), hex2num('1')=1 → byte 0x01.
    assert decode_ssid("\\x1Z") == "\x01Z"

  def test_hex_trailing_single_digit(self):
    # `\xA` at end-of-string: 2-digit fails, 1-digit succeeds → 0x0a.
    assert decode_ssid("\\xA") == "\x0a"

  def test_hex_malformed_drops_escape(self):
    # `\xGZ`: both hex2byte and hex2num fail; inner switch `break`, the
    # outer loop then emits G then Z as literals.
    assert decode_ssid("\\xGZ") == "GZ"

  def test_octal_three_digit(self):
    # `\101` = 0o101 = 65 = 'A'
    assert decode_ssid("\\101") == "A"

  def test_octal_one_digit(self):
    assert decode_ssid("\\0X") == "\x00X"

  def test_octal_stops_at_non_octal(self):
    # `\78`: '7' is octal, '8' is not → val=7, then '8' emitted as literal.
    assert decode_ssid("\\78") == "\x078"

  def test_standard_escapes(self):
    assert decode_ssid("\\\\") == "\\"
    assert decode_ssid('\\"') == '"'
    assert decode_ssid("\\n") == "\n"
    assert decode_ssid("\\r") == "\r"
    assert decode_ssid("\\t") == "\t"
    assert decode_ssid("\\e") == "\x1b"

  def test_unknown_escape_drops_backslash(self):
    # `\q`: inner default breaks without advancing pos; next iteration
    # emits 'q' as a literal. Backslash is consumed.
    assert decode_ssid("a\\qb") == "aqb"

  def test_trailing_backslash_dropped(self):
    # pos++ past '\\', then *pos == '\\0' exits the while loop.
    assert decode_ssid("abc\\") == "abc"

  def test_all_null_normalized_to_empty(self):
    # Hidden APs broadcast 32 null bytes; we normalize to "" so the
    # wifi_manager filter drops them the same as truly-empty SSIDs.
    assert decode_ssid("\\x00" * 32) == ""

  def test_partial_null_preserved(self):
    # Mixed content with an embedded NUL stays as-is (non-empty).
    assert decode_ssid("A\\x00B") == "A\x00B"


class _RacySock:
  """Stub ctrl socket that models the reply/command pairing race.

  The real wpa_supplicant ctrl socket delivers replies in send order. If two
  threads interleave their send/recv calls, the first reader sees the *other*
  thread's reply. This stub reproduces that: `send` stores the last command,
  `recv` sleeps briefly to widen the race window, then returns a reply
  based on whichever command was most recently sent.

  Under `WpaCtrl.request`'s lock, each send/recv pair is atomic, so each
  caller sees its own reply. Without the lock, thread B's send overwrites
  the stored command during thread A's sleep and A observes B's reply.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._last_sent: bytes = b""

  def send(self, data: bytes):
    with self._lock:
      self._last_sent = data

  def recv(self, _bufsize: int) -> bytes:
    # Widen the race window so any unlocked caller would mispair.
    time.sleep(0.005)
    with self._lock:
      return b"REPLY:" + self._last_sent


class TestWpaCtrlRequestSerialization:
  def test_request_pairs_reply_with_command_under_concurrency(self):
    """Regression: concurrent WpaCtrl.request callers must each observe
    the reply for their own command, not a peer's."""
    ctrl = WpaCtrl()
    ctrl._sock = _RacySock()

    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def worker(cmd: str):
      try:
        results[cmd] = ctrl.request(cmd)
      except BaseException as exc:
        errors.append(exc)

    threads = [threading.Thread(target=worker, args=(cmd,))
               for cmd in ("STATUS", "SCAN_RESULTS", "LIST_NETWORKS", "PING")]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=5)

    assert not errors, errors
    for cmd in ("STATUS", "SCAN_RESULTS", "LIST_NETWORKS", "PING"):
      assert results[cmd] == f"REPLY:{cmd}", \
        f"concurrent request mispaired reply for {cmd}: {results[cmd]}"

    # Clear the socket so __del__ doesn't try to close the stub.
    ctrl._sock = None
