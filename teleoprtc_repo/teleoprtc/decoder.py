import dataclasses
import struct
from typing import List


@dataclasses.dataclass(frozen=True)
class RtcpReceiverReport:
  ssrc: int
  fraction_lost: int
  packets_lost: int
  highest_seq_no: int
  jitter: int
  lsr: int
  dlsr: int


def _decode_receiver_reports(message: bytes) -> List[RtcpReceiverReport]:
  reports: List[RtcpReceiverReport] = []
  offset = 0

  while offset + 4 <= len(message):
    flags, packet_type, length_words = struct.unpack_from("!BBH", message, offset)
    packet_end = offset + (length_words + 1) * 4
    if flags >> 6 != 2 or packet_end > len(message):
      break

    report_count = flags & 0x1F
    if packet_type == 200:  # Sender Report
      report_offset = offset + 28
    elif packet_type == 201:  # Receiver Report
      report_offset = offset + 8
    else:
      offset = packet_end
      continue

    if report_offset + report_count * 24 > packet_end:
      break

    for i in range(report_count):
      block_offset = report_offset + i * 24
      ssrc, loss, highest_seq_no, jitter, lsr, dlsr = struct.unpack_from("!IIIIII", message, block_offset)
      fraction_lost = loss >> 24
      packets_lost = loss & 0xFFFFFF
      if packets_lost & 0x800000:
        packets_lost -= 1 << 24
      reports.append(RtcpReceiverReport(ssrc, fraction_lost, packets_lost, highest_seq_no, jitter, lsr, dlsr))

    offset = packet_end

  return reports
