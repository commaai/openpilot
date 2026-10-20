"""GET-only proxy to the few log hosts that send no CORS headers. A strict allowlist, including for redirects: never an
open relay. Nothing from the page is forwarded except Range, and no credentials are added."""
from urllib.parse import urljoin, urlparse

from openpilot.tools.lib import filereader
from openpilot.tools.lib.url_file import URLFile
from openpilot.tools.ycpe.daemon.core import HTTPError, Stream, module, route

HTTPS_HOSTS = {"commadataci.blob.core.windows.net", "huggingface.co"}
HTTPS_SUFFIXES = (".hf.co",)  # huggingface LFS redirects to its CDN (cas-bridge.xethub.hf.co, cdn-lfs.hf.co)
REDIRECTS = 5

module("proxy", "1", hosts=sorted(HTTPS_HOSTS) + [f"*{s}" for s in HTTPS_SUFFIXES] + [urlparse(filereader.DATA_ENDPOINT).hostname])


def allowed(url: str) -> bool:
  u = urlparse(url)
  if "@" in u.netloc:
    return False
  if u.scheme == "https":
    host = (u.hostname or "").lower()
    return u.port in (None, 443) and (host in HTTPS_HOSTS or host.endswith(HTTPS_SUFFIXES))
  return u.scheme == "http" and u.netloc == urlparse(filereader.DATA_ENDPOINT).netloc  # the internal endpoint, exactly


@route("GET", "/local/proxy")
def proxy(req):
  url = req.query.get("url", "")
  headers = {"Range": req.headers["range"]} if "range" in req.headers else {}
  for _ in range(REDIRECTS + 1):
    if not allowed(url):
      raise HTTPError(403, "host not allowed")
    r = URLFile.pool_manager().request("GET", url, headers=headers, preload_content=False, redirect=False, retries=False, timeout=30.0)
    if r.status in (301, 302, 303, 307, 308) and r.headers.get("location"):
      url = urljoin(url, r.headers["location"])
      r.release_conn()
      continue
    break
  else:
    raise HTTPError(502, "too many redirects")
  out = {k: r.headers[k] for k in ("Content-Range", "Accept-Ranges", "Last-Modified", "ETag") if k in r.headers}

  def gen():
    try:
      yield from r.stream(1 << 20)
    finally:
      r.release_conn()
  n = r.headers.get("content-length")
  # never the upstream's type: an HTML or SVG answer must not run as a page on this origin, which holds the token
  out["X-Upstream-Type"] = r.headers.get("content-type", "")
  out["Content-Security-Policy"] = "sandbox"
  return Stream(gen(), "application/octet-stream", out, status=r.status, length=int(n) if n else None)
