# Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
#
# This file is part of sunnypilot and is licensed under the MIT License.
# See the LICENSE.md file in the root directory for more details.

import asyncio
import os
import time

import aiohttp
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.hw import Paths

from cereal import messaging, custom
from sunnypilot.models.fetcher import ModelFetcher
from sunnypilot.models.helpers import verify_file, get_active_bundle


class ModelManagerSP:
  """Manages model downloads and status reporting"""

  def __init__(self):
    self.params = Params()
    self.model_fetcher = ModelFetcher(self.params)
    self.pm = messaging.PubMaster(["modelManagerSP"])
    self.available_models: list[custom.ModelManagerSP.ModelBundle] = []
    self.selected_bundle: custom.ModelManagerSP.ModelBundle = None
    self.active_bundle: custom.ModelManagerSP.ModelBundle = get_active_bundle(self.params)
    self._chunk_size = 128 * 1000  # 128 KB chunks
    self._download_start_times: dict[str, float] = {}  # Track start time per model

  def _calculate_eta(self, filename: str, progress: float) -> int:
    """Calculate ETA based on elapsed time and current progress"""
    if filename not in self._download_start_times or progress <= 0:
      return 60  # Default ETA for new downloads

    elapsed_time = time.monotonic() - self._download_start_times[filename]
    if elapsed_time <= 0:
      return 60

    # If we're at X% after Y seconds, we can estimate total time as (Y / X) * 100
    total_estimated_time = (elapsed_time / progress) * 100
    eta = total_estimated_time - elapsed_time

    return max(1, int(eta))  # Return at least 1 second if download is ongoing

  async def _download_file(self, url: str, path: str, model) -> None:
    """Downloads a file with progress tracking"""
    self._download_start_times[model.fileName] = time.monotonic()

    async with aiohttp.ClientSession() as session:
      async with session.get(url) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        bytes_downloaded = 0

        with open(path, 'wb') as f:
          async for chunk in response.content.iter_chunked(self._chunk_size):  # type: bytes
            f.write(chunk)
            bytes_downloaded += len(chunk)

            if total_size > 0:
              progress = (bytes_downloaded / total_size) * 100
              model.downloadProgress.status = custom.ModelManagerSP.DownloadStatus.downloading
              model.downloadProgress.progress = progress
              model.downloadProgress.eta = self._calculate_eta(model.fileName, progress)
              self._report_status()

        # Clean up start time after download completes
        del self._download_start_times[model.fileName]

  async def _process_model(self, model, destination_path: str) -> None:
    """Processes a single model download including verification"""
    url = model.downloadUri.uri
    expected_hash = model.downloadUri.sha256
    filename = model.fileName
    full_path = os.path.join(destination_path, filename)

    try:
      # Check existing file
      if os.path.exists(full_path) and await verify_file(full_path, expected_hash):
        model.downloadProgress.status = custom.ModelManagerSP.DownloadStatus.cached
        model.downloadProgress.progress = 100
        model.downloadProgress.eta = 0
        self._report_status()
        return

      # Download and verify
      await self._download_file(url, full_path, model)
      if not await verify_file(full_path, expected_hash):
        raise ValueError(f"Hash validation failed for {filename}")

      model.downloadProgress.status = custom.ModelManagerSP.DownloadStatus.downloaded
      model.downloadProgress.eta = 0
      self._report_status()

    except Exception as e:
      cloudlog.error(f"Error downloading {filename}: {str(e)}")
      if os.path.exists(full_path):
        os.remove(full_path)
      model.downloadProgress.status = custom.ModelManagerSP.DownloadStatus.failed
      model.downloadProgress.eta = 0
      self.selected_bundle.status = custom.ModelManagerSP.DownloadStatus.failed
      self._report_status()
      # Clean up start time if it exists
      self._download_start_times.pop(model.fileName, None)
      raise

  def _report_status(self) -> None:
    """Reports current status through messaging system"""
    msg = messaging.new_message('modelManagerSP', valid=True)
    model_manager_state = msg.modelManagerSP
    if self.selected_bundle:
      model_manager_state.selectedBundle = self.selected_bundle

    if self.active_bundle:
      model_manager_state.activeBundle = self.active_bundle

    model_manager_state.availableBundles = self.available_models
    self.pm.send('modelManagerSP', msg)

  async def _download_bundle(self, model_bundle: custom.ModelManagerSP.ModelBundle, destination_path: str) -> None:
    """Downloads all models in a bundle"""
    self.selected_bundle = model_bundle
    self.selected_bundle.status = custom.ModelManagerSP.DownloadStatus.downloading
    os.makedirs(destination_path, exist_ok=True)

    try:
      tasks = [self._process_model(model, destination_path)
               for model in self.selected_bundle.models]
      await asyncio.gather(*tasks)
      self.selected_bundle.status = custom.ModelManagerSP.DownloadStatus.downloaded
      self.active_bundle = self.selected_bundle
      self.params.put("ModelManager_ActiveBundle", self.selected_bundle.to_bytes())

    except Exception:
      self.selected_bundle.status = custom.ModelManagerSP.DownloadStatus.failed
      raise

    finally:
      self._report_status()

  def download(self, model_bundle: custom.ModelManagerSP.ModelBundle, destination_path: str) -> None:
    """Main entry point for downloading a model bundle"""
    asyncio.run(self._download_bundle(model_bundle, destination_path))

  def main_thread(self) -> None:
    """Main thread for model management"""
    rk = Ratekeeper(1, print_delay_threshold=None)

    while True:
      try:
        self.available_models = self.model_fetcher.get_available_models()

        if index_to_download := self.params.get("ModelManager_DownloadIndex", block=False, encoding="utf-8"):
          if model_to_download := next((model for model in self.available_models if model.index == int(index_to_download)), None):
            try:
              self.download(model_to_download, Paths.model_root())
            except Exception as e:
              cloudlog.exception(e)
            finally:
              self.params.put("ModelManager_DownloadIndex", "")

        self._report_status()
        rk.keep_time()

      except Exception as e:
        cloudlog.exception(f"Error in main thread: {str(e)}")
        rk.keep_time()


def main():
  ModelManagerSP().main_thread()


if __name__ == "__main__":
  main()
