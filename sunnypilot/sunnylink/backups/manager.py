"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import base64
import json
import time
from enum import Enum
from typing import Any

from openpilot.common.git import get_branch
from openpilot.common.params import Params, ParamKeyType
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.system.version import get_version

from cereal import messaging, custom
from sunnypilot.sunnylink.api import SunnylinkApi
from sunnypilot.sunnylink.backups.utils import decrypt_compressed_data, encrypt_compress_data, SnakeCaseEncoder


class OperationType(Enum):
  BACKUP = "backup"
  RESTORE = "restore"


class BackupManagerSP:
  """Manages device configuration backups to/from sunnylink"""

  def __init__(self):
    self.params = Params()
    self.device_id = self.params.get("SunnylinkDongleId", encoding="utf8")
    self.api = SunnylinkApi(self.device_id)
    self.pm = messaging.PubMaster(["backupManagerSP"])

    # Status tracking
    self.backup_status = custom.BackupManagerSP.Status.idle
    self.restore_status = custom.BackupManagerSP.Status.idle

    # Unified progress & operation type (only one operation runs at a time)
    self.progress = 0.0
    self.operation: OperationType | None = None

    self.last_error = ""

  def _report_status(self) -> None:
    """Reports current backup manager state through the messaging system."""
    msg = messaging.new_message('backupManagerSP', valid=True)
    backup_state = msg.backupManagerSP

    backup_state.backupStatus = self.backup_status
    backup_state.restoreStatus = self.restore_status
    # Both progress fields use the unified progress value
    backup_state.backupProgress = self.progress
    backup_state.restoreProgress = self.progress
    backup_state.lastError = self.last_error

    # Optionally, add a field for operation type if supported:
    # backup_state.operationType = self.operation.value if self.operation else "none"

    self.pm.send('backupManagerSP', msg)

  def _update_progress(self, progress: float, op_type: OperationType) -> None:
    """Updates the unified progress and operation type, then reports status."""
    self.progress = progress
    self.operation = op_type
    self._report_status()

  def _collect_config_data(self) -> dict[str, Any]:
    """Collects configuration data to be backed up."""
    config_data = {}
    params_to_backup = [k.decode('utf-8') for k in self.params.all_keys(ParamKeyType.BACKUP)]
    for param in params_to_backup:
      value = self.params.get(param)
      if value is not None:
        config_data[param] = base64.b64encode(value).decode('utf-8')
    return config_data

  def _get_metadata_value(self, metadata_list, key, default_value=None):
    return next((entry.get("value") for entry in metadata_list if entry.get("key") == key), default_value)

  async def create_backup(self) -> bool:
    """Creates and uploads a new backup to sunnylink."""
    try:
      self.backup_status = custom.BackupManagerSP.Status.inProgress
      self._update_progress(0.0, OperationType.BACKUP)

      # Collect configuration data
      config_data = self._collect_config_data()
      self._update_progress(25.0, OperationType.BACKUP)

      # Serialize and encrypt config data
      config_json = json.dumps(config_data)
      encrypted_config = encrypt_compress_data(config_json, use_aes_256=True)
      self._update_progress(50.0, OperationType.BACKUP)

      backup_info = custom.BackupManagerSP.BackupInfo()
      backup_info.deviceId = self.device_id
      backup_info.config = encrypted_config
      backup_info.isEncrypted = True
      backup_info.createdAt = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
      backup_info.updatedAt = backup_info.createdAt
      backup_info.sunnypilotVersion = self._get_current_version()
      backup_info.backupMetadata = [
        custom.BackupManagerSP.MetadataEntry(key="creator", value="BackupManagerSP"),
        custom.BackupManagerSP.MetadataEntry(key="all_values_encoded", value="True"),
        custom.BackupManagerSP.MetadataEntry(key="AES", value="256")
      ]

      payload = json.loads(json.dumps(backup_info.to_dict(), cls=SnakeCaseEncoder))
      self._update_progress(75.0, OperationType.BACKUP)

      # Upload to sunnylink
      result = self.api.api_get(
        f"backup/{self.device_id}",
        method='PUT',
        access_token=self.api.get_token(),
        json=payload
      )

      if result:
        self.backup_status = custom.BackupManagerSP.Status.completed
        self._update_progress(100.0, OperationType.BACKUP)
      else:
        self.backup_status = custom.BackupManagerSP.Status.failed
        self.last_error = "Failed to upload backup"
        self._report_status()

      return bool(self.backup_status == custom.BackupManagerSP.Status.completed)

    except Exception as e:
      cloudlog.exception(f"Error creating backup: {str(e)}")
      self.backup_status = custom.BackupManagerSP.Status.failed
      self.last_error = str(e)
      self._report_status()
      return False

  async def restore_backup(self, version: int | None = None) -> bool:
    """Restores a backup from sunnylink."""
    try:
      self.restore_status = custom.BackupManagerSP.Status.inProgress
      self._update_progress(0.0, OperationType.RESTORE)

      # Get backup data from API for the specified version
      endpoint = f"backup/{self.device_id}" + f"/{version or ''}" + "?api-version=1"
      backup_data = self.api.api_get(endpoint, access_token=self.api.get_token())
      if not backup_data:
        raise Exception(f"No backup found for device {self.device_id}")

      self._update_progress(25.0, OperationType.RESTORE)

      data = backup_data.json()
      backup_metadata = data.get("backup_metadata", [])
      encrypted_config = data.get("config", "")
      if not encrypted_config:
        raise Exception("Empty backup configuration")
      self._update_progress(50.0, OperationType.RESTORE)

      # Decrypt config and load data
      use_aes_256 = self._get_metadata_value(backup_metadata, "AES", "128") == "256"
      config_json = decrypt_compressed_data(encrypted_config, use_aes_256)
      if not config_json:
        raise Exception("Failed to decrypt backup configuration")

      config_data = json.loads(config_json)
      self._update_progress(75.0, OperationType.RESTORE)

      # Apply configuration
      all_values_encoded = self._get_metadata_value(backup_metadata, "all_values_encoded", "false")
      self._apply_config(config_data, str(all_values_encoded).lower() == "true")

      self.restore_status = custom.BackupManagerSP.Status.completed
      self._update_progress(100.0, OperationType.RESTORE)
      return True

    except Exception as e:
      cloudlog.exception(f"Error restoring backup: {str(e)}")
      self.restore_status = custom.BackupManagerSP.Status.failed
      self.last_error = str(e)
      self._report_status()
      return False

  def _apply_config(self, config_data: dict[str, str], all_values_encoded: bool = False) -> None:
    """Applies configuration data from a backup, but only for parameters marked as backupable."""
    # Get the current list of parameters that can be backed up
    backupable_params = [k.decode('utf-8') for k in self.params.all_keys(ParamKeyType.BACKUP)]

    # Count for logging/reporting
    restored_count = 0
    skipped_count = 0

    for param, encoded_value in config_data.items():
      try:
        # Only restore parameters that are currently marked as backupable
        if param in backupable_params:
          value = base64.b64decode(encoded_value) if all_values_encoded else encoded_value
          self.params.put(param, value)
          restored_count += 1
        else:
          skipped_count += 1
          cloudlog.info(f"Skipped restoring param {param}: not marked for backup in current version")
      except Exception as e:
        cloudlog.error(f"Failed to restore param {param}: {str(e)}")

    cloudlog.info(f"Restore complete: {restored_count} params restored, {skipped_count} params skipped")

  def _get_current_version(self) -> custom.BackupManagerSP.Version:
    """Gets current sunnypilot version information."""
    version_obj = custom.BackupManagerSP.Version()
    version_str = get_version()

    version_parts = version_str.split('-')  # For when version is like "1.2.3-456"
    version_nums = version_parts[0].split('.')

    # Extract build number from hyphen format or as 4th version component
    build = 0
    if len(version_parts) > 1 and version_parts[1].isdigit():
      build = int(version_parts[1])
    elif len(version_nums) > 3 and version_nums[3].isdigit():
      build = int(version_nums[3])

    # Set version components with safer defaults
    version_obj.major = int(version_nums[0]) if len(version_nums) > 0 and version_nums[0].isdigit() else 0
    version_obj.minor = int(version_nums[1]) if len(version_nums) > 1 and version_nums[1].isdigit() else 0
    version_obj.patch = int(version_nums[2]) if len(version_nums) > 2 and version_nums[2].isdigit() else 0
    version_obj.build = build
    version_obj.branch = get_branch()

    return version_obj

  async def main_thread(self) -> None:
    """Main thread for backup management."""
    rk = Ratekeeper(1, print_delay_threshold=None)
    reset_progress = False

    while True:
      try:
        if reset_progress:
          self.progress = 100.0
          self.operation = None
          self.restore_status = custom.BackupManagerSP.Status.idle
          self.backup_status = custom.BackupManagerSP.Status.idle

        # Check for backup command
        if self.params.get_bool("BackupManager_CreateBackup"):
          try:
            await self.create_backup()
            reset_progress = True
          finally:
            self.params.remove("BackupManager_CreateBackup")

        # Check for restore command
        restore_version = self.params.get("BackupManager_RestoreVersion", encoding="utf8")
        if restore_version:
          try:
            version = int(restore_version) if restore_version.isdigit() else None
            await self.restore_backup(version)
            reset_progress = True
          finally:
            self.params.remove("BackupManager_RestoreVersion")

        self._report_status()
        rk.keep_time()

      except Exception as e:
        cloudlog.exception(f"Error in backup manager main thread: {str(e)}")
        self.last_error = str(e)
        self._report_status()
        rk.keep_time()


def main():
  import asyncio
  asyncio.run(BackupManagerSP().main_thread())


if __name__ == "__main__":
  main()
