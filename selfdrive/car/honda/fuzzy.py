import re
from collections import defaultdict

from cereal import car

Ecu = car.CarParams.Ecu


# Honda Fuzzy Fingerprinting

# Terminology from Honda SIS
# - 'Programmable Control Unit' - ECU
# - 'Program ID' - firmware version

# Program ID determined format:
#   [xxxxx]-[xxx]-[xxx]
#   [classification]-[platform]-[revision]
#   - classification: 5 alphanumeric characters which represent what type of part this is, the last character varies
#                     slightly (+/-1-3), which seem to be for variations ex: for manual vs automatic transmissions
#   - platform: represents the platform that this part fits, or the first car it was used in when it fits multiple
#   - revision: represents software revision for this part. for example, one update goes from C710 to C730

HONDA_FW_PATTERN = br"(?P<classification>[A-Z0-9]{5})-(?P<platform>[A-Z0-9]{3})(-|,)(?P<revision>[A-Z0-9]{4})(\x00){2}$"

PLATFORM_CODE_ECUS = {Ecu.eps, Ecu.gateway, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.transmission, Ecu.electricBrakeBooster}
ESSENTIAL_ECUS = {Ecu.fwdRadar, Ecu.transmission, Ecu.eps, Ecu.fwdCamera}


def get_platform_codes(fw_versions: list[bytes]) -> dict[bytes, set[bytes]]:
  codes = defaultdict(set)
  for fw in fw_versions:
    m = re.match(HONDA_FW_PATTERN, fw)

    if m:
      codes[b'-'.join((m.group('classification'), m.group('platform')))].add(m.group('revision'))

  return dict(codes)


def match_fw_to_car_fuzzy(live_fw_versions, vin, offline_fw_versions) -> set[str]:
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

      # Check part number + platform code + major version matches for any found versions
      # Platform codes and major versions change for different physical parts, generation, API, etc.
      # Sub-versions are incremented for minor recalls, do not need to be checked.
      if not any(found_platform_code in expected_platform_codes for found_platform_code in found_platform_codes):
        break

      valid_found_ecus.add(addr)

    # If all live ECUs pass all checks for candidate, add it as a match
    if valid_expected_ecus.issubset(valid_found_ecus):
      candidates.add(candidate)

  return {str(c) for c in candidates}
