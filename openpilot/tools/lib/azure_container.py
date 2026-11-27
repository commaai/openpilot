import subprocess
from datetime import datetime, timedelta, UTC


class AzureContainer:
  """A blob container, accessed through the az CLI as the account from `az login`."""
  def __init__(self, account: str, container: str):
    self.ACCOUNT = account
    self.CONTAINER = container

  @property
  def BASE_URL(self) -> str:
    return f"https://{self.ACCOUNT}.blob.core.windows.net/{self.CONTAINER}/"

  def _az(self, *args: str) -> str:
    cmd = ["az", "storage", *args, "--account-name", self.ACCOUNT, "--auth-mode", "login", "--only-show-errors", "-o", "tsv"]
    return subprocess.check_output(cmd, text=True).strip()

  def sas(self) -> str:
    """A read, write and list token for the next hour."""
    expiry = (datetime.now(UTC) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")
    return self._az("container", "generate-sas", "--name", self.CONTAINER, "--as-user", "--permissions", "rwl", "--expiry", expiry)

  def has_prefix(self, prefix: str) -> bool:
    return self._az("blob", "list", "--container-name", self.CONTAINER, "--prefix", prefix, "--num-results", "1", "--query", "[].name") != ""
