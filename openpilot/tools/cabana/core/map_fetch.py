"""Bounded Overpass POST request, run through replay's Python process helper."""
import json
import sys
import urllib.parse
import urllib.request


def fetch(endpoint, query):
  if endpoint not in ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"):
    raise ValueError("Unsupported map endpoint")
  body = urllib.parse.urlencode({"data": query}).encode()
  request = urllib.request.Request(endpoint, data=body, headers={"User-Agent": "cabana-vector-map/1.0"})
  limit = 64 * 1024 * 1024
  with urllib.request.urlopen(request, timeout=30) as response:
    result = response.read(limit + 1)
  if len(result) > limit:
    raise ValueError("Map response exceeds 64 MiB")
  if not isinstance(json.loads(result), dict):
    raise ValueError("Map response must be a JSON object")
  return result


if __name__ == "__main__":
  try:
    sys.stdout.buffer.write(fetch(sys.argv[1], sys.argv[2]))
  except Exception as error:
    print(f"Map request failed: {error}", file=sys.stderr)
    raise SystemExit(1) from error
