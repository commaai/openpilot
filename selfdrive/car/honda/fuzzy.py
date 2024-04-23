import re
from collections import defaultdict

from cereal import car

Ecu = car.CarParams.Ecu


# Honda Fuzzy Fingerprinting

# Terminology from Honda SIS
# - 'Programmable Control Unit' - ECU
# - 'Program ID' - firmware version

# We currently fingerprint on the Program ID (from UDS_VERSION_REQUEST), which seems to have the following format:
#   [xxxxx]-[xxx]-[xxx]
#   [classification]-[platform]-[revision]
#   - classification: 5 alphanumeric characters which represent what type of part this is, the last character varies
#                     slightly (+/-1-3), which seem to be for variations ex: for manual vs automatic transmissions
#   - platform: a loose representation of the platform that this part fits, typically the same across a single car
#               or varying by a single character
#   - revision: seems to represents part and software revisions. for example, one update goes from C710 to C730

HONDA_FW_PATTERN = br"(?P<classification>[A-Z0-9]{5})-(?P<platform>[A-Z0-9]{3})(-|,)(?P<revision>[A-Z0-9]{4})(\x00){2}$"

PLATFORM_CODE_ECUS = {Ecu.eps, Ecu.gateway, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.transmission, Ecu.electricBrakeBooster}


def get_platform_codes(fw_versions: list[bytes]) -> dict[bytes, set[bytes]]:
  codes = defaultdict(set)
  for fw in fw_versions:
    m = re.match(HONDA_FW_PATTERN, fw)

    if m:
      codes[b'-'.join((m.group('classification'), m.group('platform')))].add(m.group('revision'))

  return dict(codes)


def match_fw_to_car_fuzzy(live_fw_versions, vin, offline_fw_versions) -> set[str]:
  # TODO: we can probably make a helper since this is extremely similar to the Toyota and Hyundai implementations
  candidates = set()

  for candidate, fws in offline_fw_versions.items():
    # Keep track of ECUs which pass all checks (platform codes, within sub-version range)
    valid_found_ecus = set()
    valid_expected_ecus = {ecu[1:] for ecu in fws if ecu[0] in PLATFORM_CODE_ECUS}
    for ecu, expected_versions in fws.items():
      addr = ecu[1:]
      # Only check ECUs expected to have platform codes
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      # Expected platform codes & versions
      expected_platform_codes = get_platform_codes(expected_versions)

      # Found platform codes & versions
      found_platform_codes = get_platform_codes(live_fw_versions.get(addr, set()))

      # Check part classification and platform code
      if not any(found_platform_code in expected_platform_codes for found_platform_code in found_platform_codes):
        break

      valid_found_ecus.add(addr)

    # If all live ECUs pass all checks for candidate, add it as a match
    if valid_expected_ecus.issubset(valid_found_ecus):
      candidates.add(candidate)

  return {str(c) for c in candidates}
