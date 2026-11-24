from http.server import BaseHTTPRequestHandler


class HTTPRequestHandler(BaseHTTPRequestHandler):
  protocol_version = "HTTP/1.1"

  def send_body(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    if headers:
      for name, value in headers.items():
        self.send_header(name, value)
    self.end_headers()
    if self.command != "HEAD":
      self.wfile.write(body)

  def log_message(self, format: str, *args: object) -> None:  # noqa: A002
    pass
