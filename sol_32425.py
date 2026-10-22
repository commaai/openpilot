```python
import random
import struct

import pytest

# Import the panda safety interface
from panda import Panda

# Import openpilot car interface and controller utilities
from openpilot.selfdrive.car import get_interface, get_radar_interface
from openpilot.selfdrive.car.car_helpers import interface_to_car
from openpilot.selfdrive.car.car_helpers import CarInterfaceBase, CarControllerBase


def _init_safety_modules(car_name: str):
    """
    Initialise both the Panda safety model and the OpenPilot safety
    callbacks for the given car. Returns a tuple of (panda, controller).
    """
    # ------------------------------------------------------------------
    # Panda side
    # ------------------------------------------------------------------
    panda = Panda()
    # The safety model name is stored on the car interface class
    # (e.g. safetyModel = "toyota")
    CarInterfaceCls = get_interface(car_name)
    panda.set_safety_model(CarInterfaceCls.safetyModel)

    # ------------------------------------------------------------------
    # OpenPilot side
    # ------------------------------------------------------------------
    # Build a dummy CarInterface instance – many fields are not needed for
    # the TX hook, but they are required for proper initialisation.
    # Use a minimal set of arguments to avoid hardware dependencies.
    CI = CarInterfaceCls(
        CarInterfaceCls.get_params(CarInterfaceCls.__name__),
        CarInterfaceCls.get_params(CarInterfaceCls.__name__).carFingerprint,
        CarInterfaceCls.get_params(CarInterfaceCls.__name__).carFingerprint,
    )
    controller = CI.CC  # CarController instance attached to the interface
    # Ensure the controller has its safety hooks wired (some models lazily set
    # them up in __init__).
    if hasattr(controller, "init"):
        controller.init(CI)

    return panda, controller


@pytest.mark.parametrize("car_name", [
    # Add car models that have known TX safety logic.
    # The list can be extended as new models are added.
    "toyota",
    "honda",
    "subaru",
    "gm",
    "ford",
])
def test_tx_message_fuzz(car_name):
    """
    Fuzz the TX safety checks for a given car model and ensure that the
    Panda implementation and the OpenPilot safety callback agree on every
    randomly generated message.
    """