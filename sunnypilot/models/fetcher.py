"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import json
import time

import requests
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from sunnypilot.models.helpers import is_bundle_version_compatible

from cereal import custom


class ModelParser:
  """Handles parsing of model data into cereal objects"""

  @staticmethod
  def _parse_download_uri(download_uri_data) -> custom.ModelManagerSP.DownloadUri:
    download_uri = custom.ModelManagerSP.DownloadUri()
    download_uri.uri = download_uri_data.get("url")
    download_uri.sha256 = download_uri_data.get("sha256")
    return download_uri

  @staticmethod
  def _parse_artifact(artifact_data) -> custom.ModelManagerSP.Artifact:
    artifact = custom.ModelManagerSP.Artifact()
    artifact.fileName = artifact_data.get("file_name")
    artifact.downloadUri = ModelParser._parse_download_uri(artifact_data.get("download_uri", {}))
    return artifact

  @staticmethod
  def _parse_model(model_data) -> custom.ModelManagerSP.Model:
    model = custom.ModelManagerSP.Model()

    model.type = model_data.get("type")
    model.artifact = ModelParser._parse_artifact(model_data.get("artifact", {}))
    if metadata := model_data.get("metadata"):
      model.metadata = ModelParser._parse_artifact(metadata)
    return model

  @staticmethod
  def _parse_bundle(bundle) -> custom.ModelManagerSP.ModelBundle:
    model_bundle = custom.ModelManagerSP.ModelBundle()
    model_bundle.index = int(bundle["index"])
    model_bundle.internalName = bundle["short_name"]
    model_bundle.displayName = bundle["display_name"]
    model_bundle.models = [ModelParser._parse_model(model) for model in bundle.get("models",[])]
    model_bundle.status = 0
    model_bundle.generation = int(bundle["generation"])
    model_bundle.environment = bundle["environment"]
    model_bundle.runner = bundle.get("runner", custom.ModelManagerSP.Runner.snpe)
    model_bundle.is20hz = bundle.get("is_20hz", False)
    model_bundle.minimumSelectorVersion = int(bundle["minimum_selector_version"])

    return model_bundle

  @staticmethod
  def parse_models(json_data: dict) -> list[custom.ModelManagerSP.ModelBundle]:
    found_bundles = [ModelParser._parse_bundle(bundle) for bundle in json_data.get("bundles", [])]
    return [bundle for bundle in found_bundles if is_bundle_version_compatible(bundle.to_dict())]


class ModelCache:
  """Handles caching of model data to avoid frequent remote fetches"""

  def __init__(self, params: Params, cache_timeout: int = int(3600 * 1e9)):
    self.params = params
    self.cache_timeout = cache_timeout
    self._LAST_SYNC_KEY = "ModelManager_LastSyncTime"
    self._CACHE_KEY = "ModelManager_ModelsCache"

  def _is_expired(self) -> bool:
    """Checks if the cache has expired"""
    current_time = int(time.monotonic() * 1e9)
    last_sync = int(self.params.get(self._LAST_SYNC_KEY, encoding="utf-8") or 0)
    return last_sync == 0 or (current_time - last_sync) >= self.cache_timeout

  def get(self) -> tuple[dict, bool]:
    """
    Retrieves cached model data and expiration status atomically.
    Returns: Tuple of (cached_data, is_expired)
    If no cached data exists or on error, returns an empty dict
    """
    try:
      cached_data = self.params.get(self._CACHE_KEY, encoding="utf-8")
      if not cached_data:
        cloudlog.warning("No cached model data available")
        return {}, True
      return json.loads(cached_data), self._is_expired()
    except Exception as e:
      cloudlog.exception(f"Error retrieving cached model data: {str(e)}")
      return {}, True

  def set(self, data: dict) -> None:
    """Updates the cache with new model data"""
    self.params.put(self._CACHE_KEY, json.dumps(data))
    self.params.put(self._LAST_SYNC_KEY, str(int(time.monotonic() * 1e9)))


class ModelFetcher:
  """Handles fetching and caching of model data from remote source"""
  MODEL_URL = "https://docs.sunnypilot.ai/driving_models_v3.json"

  def __init__(self, params: Params):
    self.params = params
    self.model_cache = ModelCache(params)
    self.model_parser = ModelParser()

  def _fetch_and_cache_models(self) -> list[custom.ModelManagerSP.ModelBundle]:
    """Fetches fresh model data from remote and updates cache"""
    try:
      response = requests.get(self.MODEL_URL, timeout=10)
      response.raise_for_status()
      json_data = response.json()

      self.model_cache.set(json_data)
      cloudlog.debug("Successfully updated models cache")
      return self.model_parser.parse_models(json_data)
    except Exception:
      cloudlog.exception("Error fetching models")
      raise

  def get_available_bundles(self) -> list[custom.ModelManagerSP.ModelBundle]:
    """Gets the list of available models, with smart cache handling"""
    cached_data, is_expired = self.model_cache.get()

    if cached_data and not is_expired:
      cloudlog.debug("Using valid cached models data")
      return self.model_parser.parse_models(cached_data)

    try:
      return self._fetch_and_cache_models()
    except Exception:
      if not cached_data:
        cloudlog.exception("Failed to fetch fresh data and no cache available")
        raise

    cloudlog.warning("Failed to fetch fresh data. Using expired cache as fallback")
    return self.model_parser.parse_models(cached_data)

if __name__ == "__main__":
  params = Params()
  model_fetcher = ModelFetcher(params)
  bundles = model_fetcher.get_available_bundles()
  for bundle in bundles:
    for model in bundle.models:
      # Print model details
      print(f"Bundle: {bundle.internalName}, Type: {model.type}, Status: {bundle.status}")
      # Print artifact details
      print(f"Artifact: {model.artifact.fileName}, Download URI: {model.artifact.downloadUri.uri}")
      # Print metadata details
      print(f"Metadata: {model.metadata.fileName}, Download URI: {model.metadata.downloadUri.uri}")
