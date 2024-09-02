from functools import wraps


def catch_exceptions(return_on_error):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      try:
        return func(*args, **kwargs)
      except Exception as e:
        # TODO: Replace print with cloudlog.error after fixing circular import issue
        print(f"Exception in {func.__name__}: {e}")
        return return_on_error
    return wrapper

  return decorator
