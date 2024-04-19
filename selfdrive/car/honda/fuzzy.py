import re
from collections import defaultdict

from cereal import car

Ecu = car.CarParams.Ecu


# Honda Fuzzy Fingerprinting

# Honda FW version format
# [12345]-[ABC]-[A123]
# [part no]-[platform]-[revision]

HONDA_PARTNO_RE = br"(?P<part_no>[A-Z0-9]{5})-(?P<platform_code>[A-Z0-9]{3})(-|,)(?P<revision>[A-Z0-9]{4})(\x00){2}$"

PLATFORM_CODE_ECUS = {Ecu.eps, Ecu.gateway, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.transmission}


def get_platform_codes(fw_versions: list[bytes]) -> dict[bytes, set[bytes]]:
  codes = defaultdict(set)  # Optional[part]-platform-major_version: set of sub_version
  for fw in fw_versions:
    m = re.match(HONDA_PARTNO_RE, fw)

    if m:
      codes[b'-'.join((m.group('part_no'), m.group('platform_code')))].add(m.group('revision'))

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
