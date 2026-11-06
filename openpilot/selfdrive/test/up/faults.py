# the faults test_faults.py injects, each is how it's injected and what openpilot and the panda have to do about it
import re
import signal
import time
from dataclasses import dataclass

from panda import Panda
from panda.python.spi import PandaSpiHandle
from openpilot.cereal import log
from openpilot.selfdrive.test.up.car import FAULTS as CAR_FAULTS, OUTAGES

# the panda resets to SILENT, pandad only sets the car's safety mode again on the next onroad
PANDA_RESET_EVENTS = ("controlsMismatch", "canError")
PANDA_RESET_FLAGS = ("hardfaultReset", "nmiReset")
# the requests that reset the panda before they return
PANDA_RESET_REQUESTS = (0xd8, 0xeb, 0xec, 0xf0)
CAN_SETTLE = 3.  # the longest CAN timeout, a slow message can still time out right after an outage


def panda_request(request: int, param: int) -> None:
  # straight to the panda, a Panda() would reset its comms under pandad
  PandaSpiHandle().controlWrite(Panda.REQUEST_OUT, request, param, 0, b'', expect_disconnect=request in PANDA_RESET_REQUESTS)


# *** injections, each one is undone by its clear() ***

class Injection:
  def inject(self, drive) -> None:
    pass

  def clear(self, drive) -> None:
    pass


@dataclass(frozen=True)
class Car(Injection):
  # the car sends a broken or faulted message, see car.py
  fault: str

  def __post_init__(self):
    assert self.fault in CAR_FAULTS, f"car.py has no {self.fault}"

  def inject(self, drive) -> None:
    drive.car.set(fault=self.fault)

  def clear(self, drive) -> None:
    drive.car.set(fault="")
    if self.fault in OUTAGES:
      drive.settled = time.monotonic() + CAN_SETTLE


@dataclass(frozen=True)
class PandaDebug(Injection):
  # a panda debug request, see panda's board/main_comms.h
  request: int
  param: int = 0
  undo: bool = False  # the same request again right away flips it back

  def inject(self, drive) -> None:
    panda_request(self.request, self.param)
    if self.undo:
      time.sleep(0.05)
      panda_request(self.request, self.param)


@dataclass(frozen=True)
class Process(Injection):
  # a signal to one of openpilot's processes
  name: str
  sig: signal.Signals

  def inject(self, drive) -> None:
    drive.signal(self.name, self.sig)

  def clear(self, drive) -> None:
    if self.sig == signal.SIGSTOP:
      drive.signal(self.name, signal.SIGCONT)


class IgnitionOff(Injection):
  def inject(self, drive) -> None:
    drive.up.set_ignition(False)


class PandaReset(Injection):
  # resets the panda once the fault is cleared, for the faults that latch in it
  def clear(self, drive) -> None:
    panda_request(0xd8, 0)


# *** what openpilot has to do ***

@dataclass(frozen=True)
class Disengages:
  events: frozenset[str]  # one of these has to show up, any if empty
  within: float  # seconds from the fault to disengaged
  soft: bool = False   # through softDisabling
  panda_drop: bool = True  # the panda drops controls_allowed ~3 s after pandad's disengaged heartbeat


def disengages(*events: str, within: float, **kwargs) -> Disengages:
  return Disengages(frozenset(events), within, **kwargs)


@dataclass(frozen=True)
class Fault:
  name: str
  inject: tuple[Injection, ...]  # in order
  expect: Disengages | None = None  # None if openpilot stays engaged through it
  report: str = ""  # pandaStates fault or flag openpilot has to report
  also_reports: tuple[str, ...] = ()  # other faults the panda raises on the same injection
  log: tuple[str, ...] = ()  # regexes of panda debug log lines, pandad forwards them in logMessage
  latched: bool = False  # needs an ignition cycle to recover
  touch: bool = True     # a wheel touch turns the EPS faults into warnings
  breaks_rates: bool = False  # breaks the timer the panda's interrupt rate checks run on, so they can fire after it

  @property
  def offroad(self) -> bool:
    return any(isinstance(i, IgnitionOff) for i in self.inject)

  @property
  def resets_panda(self) -> bool:
    return self.report in PANDA_RESET_FLAGS


def car(name: str, expect: Disengages, **kwargs) -> Fault:
  return Fault(name, (Car(name),), expect, **kwargs)


def panda(name: str, request: int, param: int, report: str = "", log: tuple[str, ...] = (), **kwargs) -> Fault:
  # a panda fault openpilot reports and stays engaged through
  return Fault(name, (PandaDebug(request, param),), report=report, log=log, **kwargs)


def panda_reset(name: str, request: int, flag: str, log: str) -> Fault:
  return Fault(name, (PandaDebug(request, 0),), disengages(*PANDA_RESET_EVENTS, within=1.0), report=flag, log=(log,), latched=True)


def fault_bit(report: str) -> str:
  return f"{1 << log.PandaState.FaultType.schema.enumerants[report]:08x}"


def telemetry(report: str, file: str, value: str = "[0-9a-f]{8}") -> str:
  # printed by fault_occurred_value(), with the call site
  return rf"Fault 0x{fault_bit(report)} at \S*{re.escape(file)}:\d+ value 0x{value}"


def fault_occurred(report: str) -> str:
  return f"fault occurred: 0x{fault_bit(report)}"


# each group shares a panda boot and ends with a panda reset, faults that break the tick or a timer go last in it
FAULTS = [
  panda("panda_relay_readback", 0xf5, 0, "outputReadback", (fault_occurred("outputReadback"),)),
  panda("panda_fdcan_config", 0xf5, 4, "registerDivergent", ("Register 0x4000a054 divergent",)),  # FDCAN1 IE
  panda("panda_vrefint", 0xf5, 2, "adc", (telemetry("adc", "lladc.h"),)),  # AVDD from VREFINT, mV
  panda("panda_dma", 0xed, 5, "dma", (telemetry("dma", "self_test.h"),)),  # a short DMA copy, which words are wrong
  panda("panda_flash_ecc", 0xed, 4, "ecc", (telemetry("ecc", "self_test.h"),)),  # the flash ECC fail address
  # TIM6_DAC, the IPSR
  panda("panda_interrupt_timer_flag", 0xef, 6, "unusedInterruptHandled", (telemetry("unusedInterruptHandled", "interrupts.h", "00000046"),)),
  panda("panda_interrupt_timer", 0xf5, 1, "timer", (telemetry("timer", "interrupts.h"),), breaks_rates=True),  # us per two periods
  panda("panda_interrupt_timer_missing", 0xef, 1, "interruptMissing", (fault_occurred("interruptMissing"),), breaks_rates=True),
  car("pcm_cancel", disengages("pcmDisable", within=0.5)),
  panda_reset("panda_hardfault", 0xeb, "hardfaultReset", r"HardFault CFSR 0x00010000 address 0x[0-9a-f]{8} PC 0x[0-9a-f]{8}"),  # UNDEFINSTR

  panda("panda_clock", 0xef, 3, "clock", (telemetry("clock", "clock.h"),)),  # timer counts per LSI capture
  panda("panda_unused_peripheral", 0xf5, 5, "registerDivergent", ("Register 0x580244e8 divergent",)),  # RCC APB1LENR
  panda("panda_tick_timer", 0xf5, 3, "timer", (telemetry("timer", "main.c"),)),
  car("brake_press", disengages("pedalPressed", within=0.5)),
  car("can_bad_checksum_eps", disengages("canError", within=1.5)),
  car("eps_perm_fault", disengages("steerUnavailable", within=0.5, panda_drop=False), touch=False),
  panda_reset("panda_nmi", 0xec, "nmiReset", r"NMI PC 0x[0-9a-f]{8}"),

  car("can_all_silent", disengages("canBusMissing", "canError", within=1.0, panda_drop=False)),
  car("acc_fault", disengages("accFaulted", within=0.5, panda_drop=False)),
  panda("panda_nvic_priority", 0xef, 5, "registerDivergent", (telemetry("registerDivergent", "interrupts.h", "00000100"),)),  # 0x100 + the IRQ
  panda("panda_tick_missing", 0xef, 0, "interruptMissing", (fault_occurred("interruptMissing"),)),
  # MMARVALID, MSTKERR and DACCVIOL, stacked into the guard
  panda_reset("panda_stack_overflow", 0xf0, "hardfaultReset", r"HardFault CFSR 0x00000092 address 0x[0-9a-f]{8} PC 0x00000000"),

  # the flash CRC pass is ~10.6 s and only fails at its end, right after a panda reset is the shortest wait
  panda("panda_self_test_flash", 0xed, 2, "selfTestFlash", (telemetry("selfTestFlash", "self_test.h"),)),  # the computed CRC
  panda("panda_self_test_cpu", 0xed, 0, "selfTestCpu", (fault_occurred("selfTestCpu"),)),
  panda("panda_self_test_ram", 0xed, 1, "selfTestRam", (telemetry("selfTestRam", "self_test.h"),)),  # the failing block
  panda("panda_ecc", 0xed, 3, "ecc", (telemetry("ecc", "self_test.h"),)),  # the RAM and word with the ECC error
  panda("panda_unused_interrupt", 0xef, 2, "unusedInterruptHandled", (telemetry("unusedInterruptHandled", "main.c", "0000003b"),)),  # the IPSR
  # controls_allowed written outside the safety hooks and flipped back before selfdrived sees a mismatch,
  # the redundant safety state catches it too. the log has the changed address
  Fault("panda_mpu", (PandaDebug(0xee, 0, undo=True),), report="mpu", also_reports=("safetyRedundancy",), log=(telemetry("mpu", "llmpu.h"),)),
  # an unused IRQ enabled outside nvic_enable
  panda("panda_nvic_enable", 0xef, 4, "registerDivergent", (telemetry("registerDivergent", "interrupts.h", "00000000"),)),  # the IRQ
  # the SBU readings are only used and cross-checked while the relay is released, so the ignition goes off.
  # value is SBU1 in mV on ADC1 << 16 | on ADC2
  Fault("panda_adc_cross_check", (PandaDebug(0xf5, 6), IgnitionOff(), PandaReset()), disengages(within=6.0), report="adc",
        log=(telemetry("adc", "harness.h"),), latched=True),

  # every other group already sets registerDivergent
  panda("panda_mpu_region", 0xf5, 7, "registerDivergent", (telemetry("registerDivergent", "llmpu.h"),)),  # an enabled unused region
  # set_canfd_auto on a bus that doesn't exist, mustn't fault
  panda("panda_canfd_auto_bus", 0xe8, 0xffff),
  Fault("ignition_off", (IgnitionOff(),), disengages(within=6.0), latched=True),
  Fault("kill_controlsd", (Process("controlsd", signal.SIGKILL),), disengages("commIssue", "processNotRunning", within=4.0, soft=True), latched=True),
  car("can_bus0_silent", disengages("canBusMissing", "canError", within=1.0, panda_drop=False)),
  car("eps_temp_fault", disengages("steerTempUnavailable", within=4.0, soft=True, panda_drop=False), touch=False),
  # openpilot is only engageable again ~15 s after modeld resumes, so it's last
  Fault("stop_modeld", (Process("modeld", signal.SIGSTOP),), disengages("commIssue", "modeldLagging", "cameraMalfunction", within=4.5, soft=True,
                                                                         panda_drop=False)),
]
