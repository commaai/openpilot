NM_KEYFILE_ESCAPES = {
  "\\": "\\",
  "n": "\n",
  "r": "\r",
  "s": " ",
  "t": "\t",
}


def decode_nm_keyfile_string(value: str) -> str:
  decoded = []
  i = 0
  while i < len(value):
    if value[i] == "\\" and i + 1 < len(value):
      escaped = NM_KEYFILE_ESCAPES.get(value[i + 1])
      if escaped is not None:
        decoded.append(escaped)
        i += 2
        continue
    decoded.append(value[i])
    i += 1
  return "".join(decoded)


def decode_nm_keyfile_ssid(ssid: str) -> str:
  ssid = decode_nm_keyfile_string(ssid)
  if r"\;" in ssid:
    return ssid.replace(r"\;", ";")
  if not ssid.endswith(";"):
    return ssid

  try:
    ssid_bytes = bytes(int(p) for p in ssid[:-1].split(";"))
  except ValueError:
    return ssid

  if not ssid_bytes:
    return ssid
  if all(b == 0 for b in ssid_bytes):
    return ""
  return ssid_bytes.decode("utf-8", errors="surrogateescape")
