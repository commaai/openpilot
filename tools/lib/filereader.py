import os

if "COMMA_PARALLEL_DOWNLOADS" in os.environ:
  from tools.lib.url_file_parallel import URLFileParallel as URLFile
else:
  from tools.lib.url_file import URLFile  # type: ignore


def FileReader(fn, debug=False):
  if fn.startswith("http://") or fn.startswith("https://"):
    return URLFile(fn, debug=debug)
  else:
    return open(fn, "rb")
