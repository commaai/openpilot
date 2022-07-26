import asyncio


def create_periodic_task(task, frequency=1):
  async def periodic_task():
    while True:
      task()
      await asyncio.sleep(1 / frequency)

  return asyncio.create_task(periodic_task())
