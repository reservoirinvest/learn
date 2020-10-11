# Program to show that awaitables are regular objects!
# Ref: [EdgeDB youtube](https://youtu.be/-CzqsgaXUM8?t=710)

import asyncio
import datetime


def print_now():
    print(datetime.datetime.now())


async def keep_printing(name: str = "") -> None:
    while True:
        print(name, end=" ")
        print_now()
        await asyncio.sleep(0.5)


async def async_main() -> None:
    kp = keep_printing("Hey")
    waiter = asyncio.wait_for(kp, 3)
    try:
        waiter  # oops, forgot `await`!
    except asyncio.TimeoutError:
        print("oops, time's up again!")


asyncio.run(async_main())

# Using $ PYTHONASNCIODEBUG=1 gives the debug warning
# Using # PYTHONTRACEMALLOC=1 gives more details
