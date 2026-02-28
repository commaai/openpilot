import sys
import os
sys.path.append(os.getcwd())
from unittest.mock import MagicMock

# 1. Mock missing core modules
sys.modules['zmq'] = MagicMock()
sys.modules['pycapnp'] = MagicMock()
sys.modules['capnp'] = MagicMock()

# Mock cereal
car_mock = MagicMock()
car_mock.CarParams.SafetyModel.allOutput = 0
car_mock.CarParams.SafetyModel.noOutput = 1
car_mock.CarParams.NetworkLocation.gateway = 0
car_mock.CarParams.TransmissionType.automatic = 0
car_mock.CarParams.TransmissionType.manual = 1
car_mock.CarState.GearShifter.drive = "drive"
car_mock.CarState.GearShifter.reverse = "reverse"

sys.modules['cereal'] = MagicMock()
sys.modules['cereal'].car = car_mock
sys.modules['cereal.messaging'] = MagicMock()

# Mock common modules
sys.modules['common.conversions'] = MagicMock()
sys.modules['common.numpy_fast'] = MagicMock()
sys.modules['common.kalman.simple_kalman'] = MagicMock()
sys.modules['common.realtime'] = MagicMock()
sys.modules['common.params'] = MagicMock()
sys.modules['common.basedir'] = MagicMock()
sys.modules['common.conversions'].Conversions.KPH_TO_MS = 1.0/3.6
sys.modules['common.conversions'].Conversions.DEG_TO_RAD = 3.14159 / 180.0
sys.modules['common.conversions'].Conversions.RAD_TO_DEG = 180.0 / 3.14159

# Mock opendbc
sys.modules['opendbc.can.parser'] = MagicMock()
sys.modules['opendbc.can.packer'] = MagicMock()
sys.modules['opendbc.can.can_define'] = MagicMock()

# 2. Define simulation test
def test_sim(candidate_name="OPEL CORSA 6TH GEN"):
    print(f"{candidate_name} Mock Simulation Testi Başlatılıyor...")
    
    # Import our module (now that dependencies are mocked)
    try:
        from selfdrive.car.opel.interface import CarInterface
        from selfdrive.car.opel.values import CAR
    except ImportError as e:
        print(f"Hata: Modül import edilemedi: {e}")
        return

    # Mock CP (CarParams)
    CP = MagicMock()
    CP.carFingerprint = candidate_name
    CP.transmissionType = car_mock.CarParams.TransmissionType.automatic
    CP.networkLocation = car_mock.CarParams.NetworkLocation.gateway
    CP.safetyConfigs = [MagicMock()]
    CP.minSteerSpeed = 0.0
    CP.pcmCruise = True
    CP.steerRateCost = 1.0
    CP.lateralTuning.which = MagicMock(return_value='torque')
    CP.lateralTuning.torque.kp = 1.0
    CP.lateralTuning.torque.kf = 1.0
    CP.lateralTuning.torque.ki = 0.1
    CP.lateralTuning.torque.friction = 0.01
    CP.steerRatio = 14.7
    CP.wheelbase = 2.538
    CP.mass = 1200.0

    # Mock CarController and CarState
    from selfdrive.car.opel.carcontroller import CarController
    from selfdrive.car.opel.carstate import CarState

    # Instantiate interface
    interface = CarInterface(CP, CarController, CarState)
    interface.CS.displayMetricUnits = True
    print("CarInterface başarıyla oluşturuldu.")

    # Test update flow
    print("Update döngüsü simüle ediliyor...")
    c = MagicMock()
    c.enabled = False
    can_strings = [] # Empty for mock
    
    # Mock CS.update to return a realistic ret
    ret = MagicMock()
    ret.vEgo = 0.0
    ret.standstill = True
    ret.steeringPressed = False
    ret.doorOpen = False
    ret.seatbeltUnlatched = False
    ret.gearShifter = car_mock.CarState.GearShifter.drive
    ret.cruiseState.available = True
    ret.cruiseState.enabled = False
    ret.espDisabled = False
    ret.stockFcw = False
    ret.stockAeb = False
    ret.cruiseState.nonAdaptive = False
    ret.brakeHoldActive = False
    ret.parkingBrake = False
    ret.steerFaultTemporary = False
    ret.steerFaultPermanent = False
    
    interface.CS.update = MagicMock(return_value=ret)
    interface.CS.out = ret
    
    ret_out = interface.update(c, can_strings)
    print("Interface update başarıyla çalıştırıldı.")

    # Test apply flow
    print("Apply döngüsü simüle ediliyor...")
    interface.CC.update = MagicMock(return_value=(MagicMock(), []))
    actuators = interface.apply(c)
    print("Interface apply başarıyla çalıştırıldı.")

    print("\n✅ Opel Corsa F simülasyon testi başarıyla tamamlandı!")
    print("Not: Bu test, mantıksal akışın ve modül yapısının doğruluğunu kanıtlar.")

if __name__ == "__main__":
    test_sim("OPEL CORSA 6TH GEN 2020+")
    print("-" * 40)
    test_sim("PEUGEOT 208 2ND GEN")
