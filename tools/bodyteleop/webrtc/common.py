import aiortc
import asyncio
import aiohttp

import dataclasses
import json

from typing import List, Callable, Awaitable


@dataclasses.dataclass
class StreamingOffer:
  sdp: str
  type: str
  video: List[str]
  audio: bool
  channel: bool


ConnectionProvider = Callable[[StreamingOffer], Awaitable[aiortc.RTCSessionDescription]]


class StdioConnectionProvider:
  async def __call__(self, offer: StreamingOffer) -> aiortc.RTCSessionDescription:
    async def async_input():
      return await asyncio.to_thread(input)

    print("-- Please send this JSON to server --")
    print(json.dumps(dataclasses.asdict(offer)))
    print("-- Press enter when the answer is ready --")
    raw_payload = await async_input()
    payload = json.loads(raw_payload)
    answer = aiortc.RTCSessionDescription(**payload)

    return answer


class HTTPConnectionProvider:
  def __init__(self, address="127.0.0.1", port=8080):
    self.address = address
    self.port = port

  async def __call__(self, offer: StreamingOffer) -> aiortc.RTCSessionDescription:
    payload = dataclasses.asdict(offer)
    async with aiohttp.ClientSession() as session:
      response = await session.get(f"http://{self.address}:{self.port}/webrtc", json=payload)
      async with response:
        if response.status != 200:
          raise Exception(f"Offer request failed with HTTP status code {response.status}")
        answer = await response.json()
        remote_offer = aiortc.RTCSessionDescription(sdp=answer.sdp, type=answer.type)

        return remote_offer
