import platform


def suffix():
  if platform.system() == "Darwin":
    return ".dylib"
  else:
    return ".so"
