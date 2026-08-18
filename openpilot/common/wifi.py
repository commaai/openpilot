import os


WPA_CTRL_DIR = "/run/openpilot-wpa"
WPA_CTRL_PATH = os.path.join(WPA_CTRL_DIR, "wlan0")
WPA_PID_FILE = os.path.join(WPA_CTRL_DIR, "wpa_supplicant.pid")

_HEX = "0123456789abcdefABCDEF"


def decode_wpa_ssid(encoded: str) -> str:
  """Decode a wpa_supplicant printf_encode'd SSID without losing byte identity."""
  out = bytearray()
  i = 0
  while i < len(encoded):
    char = encoded[i]
    if char != "\\":
      out.append(ord(char) & 0xff)
      i += 1
      continue

    i += 1
    if i >= len(encoded):
      break

    escaped = encoded[i]
    if escaped == "\\":
      out.append(ord("\\"))
      i += 1
    elif escaped == '"':
      out.append(ord('"'))
      i += 1
    elif escaped == "n":
      out.append(ord("\n"))
      i += 1
    elif escaped == "r":
      out.append(ord("\r"))
      i += 1
    elif escaped == "t":
      out.append(ord("\t"))
      i += 1
    elif escaped == "e":
      out.append(0x1b)
      i += 1
    elif escaped == "x":
      i += 1
      if i + 1 < len(encoded) and encoded[i] in _HEX and encoded[i + 1] in _HEX:
        out.append(int(encoded[i:i + 2], 16))
        i += 2
      elif i < len(encoded) and encoded[i] in _HEX:
        out.append(int(encoded[i], 16))
        i += 1
    elif "0" <= escaped <= "7":
      value = ord(escaped) - ord("0")
      i += 1
      if i < len(encoded) and "0" <= encoded[i] <= "7":
        value = value * 8 + ord(encoded[i]) - ord("0")
        i += 1
        if i < len(encoded) and "0" <= encoded[i] <= "7":
          value = value * 8 + ord(encoded[i]) - ord("0")
          i += 1
      out.append(value & 0xff)
    # Unknown escapes consume only the backslash.

  if not out or all(byte == 0 for byte in out):
    return ""
  return out.decode("utf-8", errors="surrogateescape")
