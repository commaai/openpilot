import asyncio

BYE = 123

class Signaling:
  async def connect(self):
    pass
    return {"is_initiator": "true"}
  async def close(self):
    pass
  async def send(self, msg):
    print(msg)
    pass
  async def receive(self):
    await asyncio.sleep(1)
    pass
