import os

dir_cache: dict[str, 'CachedDir'] = {}


class CachedDir:
  def __init__(self, path: str, listing: list[str], mtime: float):
    self.path = path
    self.listing = listing
    self.mtime = mtime

  def is_fresh(self) -> bool:
    """Check if the cached listing is still valid."""
    current_mtime = os.path.getmtime(self.path)
    return current_mtime == self.mtime

  @staticmethod
  def listdir(path: str) -> list[str]:
    """
    Return cached directory listing if the directory's mtime hasn't changed,
    otherwise re-fetch and cache it.
    """
    # Check if the path is already cached and still fresh
    if path in dir_cache and dir_cache[path].is_fresh():
      return dir_cache[path].listing

    # If not cached or stale, fetch the listing
    listing: list[str] = os.listdir(path)
    dir_cache[path] = CachedDir(path, listing, os.path.getmtime(path))
    return listing
