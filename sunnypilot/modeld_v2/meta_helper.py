from openpilot.sunnypilot.modeld_v2.constants import Meta
from cereal import custom
from openpilot.sunnypilot.modeld_v2.meta_20hz import Meta20hz
from openpilot.sunnypilot.models.helpers import get_active_bundle

ModelBundle = custom.ModelManagerSP.ModelBundle


def load_meta_constants():
  """
  Determines and loads the appropriate meta model class based on the metadata provided. The function checks
  specific keys and conditions within the provided metadata dictionary to identify the corresponding meta
  model class to return.

  :param model_metadata: Dictionary containing metadata about the model. It includes
      details such as input shapes, output slices, and other configurations for identifying
      metadata-dependent meta model classes.
  :type model_metadata: dict
  :return: The appropriate meta model class (Meta, MetaSimPose, or MetaTombRaider)
      based on the conditions and metadata provided.
  :rtype: type
  """
  if (bundle := get_active_bundle()) and bundle.is20hz:
    return Meta20hz

  return Meta  # Default
