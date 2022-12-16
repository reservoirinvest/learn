# does not work after python 3.8
import asyncio

import time

def regular_function():
    time.sleep(3)
    return 0

@asyncio.coroutine
def async_function():
    yield from asyncio.sleep(3)
    return 0


regular_function()