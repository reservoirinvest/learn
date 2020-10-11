# Concurrently running coros with gather

# wait_for raises a timeout error which cancels
# if gather() is canceled, all submitted awaitables (not yet completed) are also cancelled.

import asyncio
import datetime


def print_now():
    print(datetime.datetime.now())


async def keep_printing(name: str = "") -> None:
    while True:
        print(name, end=" ")
        print_now()
        # upon cancellation, an Exception is raised here.
        await asyncio.sleep(0.5)
        # there are two choices:
        # a) deal with cancellation here with try/except block at this point
        # b) let it raise the error and let the caller / gatherer deal with the exception.

# Let us use asyncio.wait_for() to deal with the keyboard interrupt.


async def async_coro_gather_wait() -> None:
    try:
        # We have just one await here....
        # ... which wraps a gather coroutine
        # ... which gathers 3 printing coroutines
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


async def keep_printing_x(name: str = "") -> None:
    while True:
        print(name, end=" ")
        print_now()
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            print(name, "printing was cancelled!")
            break


async def async_coro_gather_wait_w_cancel() -> None:
    try:
        # We have just one await here....
        # ... which wraps a gather coroutine
        # ... which gathers 3 printing coroutines
        await asyncio.wait_for(
            asyncio.gather(
                keep_printing_x("First"),
                keep_printing_x("Second"),
                keep_printing_x("Third")
            ),
            timeout=3
        )
    except asyncio.TimeoutError:
        print("Coroutine timed out!!")

asyncio.run(async_coro_gather_wait_w_cancel())

# asyncio still complains that the cancel of gather itself was not handled by us.
# this can be solved with task and futures - in the asyncio framework.

# Coroutines are found in collections.abc alongside its parent, the awaitables. They are core features of python built into the interpreter.
# note that if you use twisted, tornado, curio, ... or any other framework, you will not use asyncio's futures and tasks.
# cororoutine is a low-level block which doesn't know about asyncio concept like event loop or cancellations.
# that is why asyncio functions are wrapped in `tasks`
