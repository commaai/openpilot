import time
import functools


def retry(attempts=3, delay=1.0, ignore_failure=False):
  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      for _ in range(attempts):
        try:
          return func(*args, **kwargs)
        except Exception:
          from openpilot.common.swaglog import cloudlog
          cloudlog.exception(f"{func.__name__} failed, trying again")
          time.sleep(delay)

      if ignore_failure:
        from openpilot.common.swaglog import cloudlog
        cloudlog.error(f"{func.__name__} failed after retry")
      else:
        raise Exception(f"{func.__name__} failed after retry")
    return wrapper
  return decorator


if __name__ == "__main__":
  @retry(attempts=10)
  def abc():
    raise ValueError("abc failed :(")
  abc()
