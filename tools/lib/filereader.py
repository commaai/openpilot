from tools.lib.url_file import URLFile

def FileReader(fn, debug=False):
  if fn.startswith("http://") or fn.startswith("https://"):
    return URLFile(fn, debug=debug)
  return open(fn, "rb")
