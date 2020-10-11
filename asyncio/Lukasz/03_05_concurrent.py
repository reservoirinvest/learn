# Concurrently running coros

import asyncio
import datetime


def print_now():
    print(datetime.datetime.now())


async def keep_printing(name: str = "") -> None:
    while True:
        print(name, end=" ")
        print_now()
        await asyncio.sleep(0.5)

# Let us use asyncio.wait_for() to deal with the keyboard interrupt.


async def async_coro_gather_wait() -> None:
    try:
        await asyncio.wait_for(
            asyncio.gather(
                keep_printing("First"),
                keep_printing("Second"),
                keep_printing("Third")
            ),
            timeout=3
        )
    except asyncio.TimeoutError:
        print("Coroutine timed out!!")

asyncio.run(async_coro_gather_wait())

# Still this complains that the _GatheringFuture exception was never retrieved!
# keep_printing coroutine raised a CancelledError exception, but we never retrieved it.
