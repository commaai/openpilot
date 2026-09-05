NM_KEYFILE_ESCAPES = {
  "\\": "\\",
  "n": "\n",
  "r": "\r",
  "s": " ",
  "t": "\t",
}


def _decode_nm_keyfile_string(value: str, *, semicolon_escape: bool) -> str:
  decoded = []
  i = 0
  while i < len(value):
    if value[i] == "\\" and i + 1 < len(value):
      escaped = NM_KEYFILE_ESCAPES.get(value[i + 1])
      if escaped is None and semicolon_escape and value[i + 1] == ";":
        escaped = ";"
      if escaped is not None:
        decoded.append(escaped)
        i += 2
        continue
    decoded.append(value[i])
    i += 1
  return "".join(decoded)


def decode_nm_keyfile_string(value: str) -> str:
  return _decode_nm_keyfile_string(value, semicolon_escape=False)


def decode_nm_keyfile_ssid(ssid: str) -> str:
  # Netplan uses a semicolon-terminated decimal byte list for SSIDs that do not
  # round-trip as a plain keyfile string. Detect that representation before
  # interpreting any keyfile escapes.
  if ssid.endswith(";"):
    parts = ssid[:-1].split(";")
    if parts and all(part.isascii() and part.isdigit() for part in parts):
      try:
        ssid_bytes = bytes(int(part) for part in parts)
      except ValueError:
        pass
      else:
        if all(byte == 0 for byte in ssid_bytes):
          return ""
        return ssid_bytes.decode("utf-8", errors="surrogateescape")

  return _decode_nm_keyfile_string(ssid, semicolon_escape=True)
