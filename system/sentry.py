"""Install exception handler for process crash."""
import sentry_sdk
from enum import Enum
from sentry_sdk.integrations.threading import ThreadingIntegration

from openpilot.common.params import Params
from openpilot.system.athena.registration import is_registered_device
from openpilot.system.hardware import HARDWARE, PC
from openpilot.common.swaglog import cloudlog
from openpilot.system.version import get_build_metadata, get_version


class SentryProject(Enum):
  # All sentry data goes to AF. comma doesn't want fork data anyway
  # python project
  SELFDRIVE = "https://741d934bafba6e550d348f17c32dc1dc@o1269754.ingest.sentry.io/4506477324533760"
  # native project
  SELFDRIVE_NATIVE = "https://e2e617f1ccebd58ed7aa8fbb15ed5b95@o1269754.ingest.sentry.io/4506477325582336"
  # controlsd CP
  AF = "https://21654306f32a4cc29d283e7e068cf27c@o1269754.ingest.sentry.io/6460006"


def capture_message(msg: str, data: str, file: str) -> None:
  try:
    dongle = Params().get("DongleId", encoding='utf-8')
    # Encode bytes to base64 for attachment
    attachment = str(data).encode()
    # Add attachment to the current scope
    # with sentry_sdk.start_transaction() as transaction:
    with sentry_sdk.configure_scope() as scope:
      scope.add_attachment(
          bytes=attachment,
          filename=f'{dongle}_{file}.txt',
          content_type="text/plain"
      )
    sentry_sdk.capture_message(f'{dongle}: {msg}')
    sentry_sdk.flush()  # https://github.com/getsentry/sentry-python/issues/291
  except Exception as e:
    print(e)

def report_tombstone(fn: str, message: str, contents: str) -> None:
  cloudlog.error({'tombstone': message})

  with sentry_sdk.configure_scope() as scope:
    scope.set_extra("tombstone_fn", fn)
    scope.set_extra("tombstone", contents)
    sentry_sdk.capture_message(message=message)
    sentry_sdk.flush()


def capture_exception(*args, **kwargs) -> None:
  cloudlog.error("crash", exc_info=kwargs.get('exc_info', 1))

  try:
    sentry_sdk.capture_exception(*args, **kwargs)
    sentry_sdk.flush()  # https://github.com/getsentry/sentry-python/issues/291
  except Exception:
    cloudlog.exception("sentry exception")


def set_tag(key: str, value: str) -> None:
  sentry_sdk.set_tag(key, value)


def init(project: SentryProject) -> bool:
  build_metadata = get_build_metadata()
  if not is_registered_device() or PC:
    print('sentry: device or remote not allowed')
    return False

  env = "release" if build_metadata.tested_channel else "master"
  dongle_id = Params().get("DongleId", encoding='utf-8')

  integrations = []
  if project == SentryProject.SELFDRIVE:
    integrations.append(ThreadingIntegration(propagate_hub=True))

  sentry_sdk.init(project.value,
                  default_integrations=False,
                  release=get_version(),
                  integrations=integrations,
                  traces_sample_rate=1.0,
                  max_value_length=8192,
                  environment=env)

  build_metadata = get_build_metadata()

  sentry_sdk.set_user({"id": dongle_id})
  sentry_sdk.set_tag("dirty", build_metadata.openpilot.is_dirty)
  sentry_sdk.set_tag("origin", build_metadata.openpilot.git_normalized_origin)
  sentry_sdk.set_tag("branch", build_metadata.channel)
  sentry_sdk.set_tag("commit", build_metadata.openpilot.git_commit)
  sentry_sdk.set_tag("device", HARDWARE.get_device_type())

  if project == SentryProject.SELFDRIVE:
    sentry_sdk.Hub.current.start_session()

  return True
