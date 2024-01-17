from openpilot.tools.lib.openpilotcontainers import OpenpilotCIContainer


def get_url(*args, **kwargs):
  return OpenpilotCIContainer.get_url(*args, **kwargs)

def upload_file(*args, **kwargs):
  return OpenpilotCIContainer.upload_file(*args, **kwargs)