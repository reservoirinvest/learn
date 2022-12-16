# running coroutines in a sequence, one after another

import asyncio
import datetime
import inspect


def print_now():
    print(datetime.datetime.now())


async def keep_printing(name: str = "") -> None:
    while True:
        print(name, end=" ")
        print_now()
        await asyncio.sleep(0.5)


async def async_main() -> None:
    await keep_printing("First")
    await keep_printing("Second")
    await keep_printing("Third")

# this simply won't run Second or Third, as the first one keeps getting printed.
# asyncio.run(async_main())

# if ALL have to be run concurrently, we can use one await with gather coroutine


async def async_coro_gather() -> None:
    await asyncio.gather(
        keep_printing("First"),
        keep_printing("Second"),
        keep_printing("Third")
    )

# asyncio.run(async_coro_gather()) # This starts all the three at almost the same time, but sequentially

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
