# Difference between coroutines and async functions
# Ref: [EdgeDB youtube](https://youtu.be/-CzqsgaXUM8?t=806)

import asyncio
import datetime


def print_now():
    print(datetime.datetime.now())


async def print3times() -> None:
    for _ in range(3):
        print_now()
        await asyncio.sleep(0.1)


coro1 = print3times()
coro2 = print3times()

print(type(print3times))
print('\n')

print(type(coro1))
print('\n')

print(type(coro2))

# The following will not work, as asyncio.run expects a coroutine! It got a function instead !!
# asyncio.run(print3times)

asyncio.run(coro1)
print('\n')
asyncio.run(coro2)

# Re-running back the coroutine already awaited, will also not work, as coroutine cannot be resued!
# Coroutines can only be awaited on once.

asyncio.run(coro1)
