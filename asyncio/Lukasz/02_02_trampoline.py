# Trampoline program schedules itself back into the event loop after running
# Ref: [Trampoline function](https://youtu.be/E7Yn5biBZ58?t=337)


import datetime
import asyncio


def print_now():
    print(datetime.datetime.now())


def trampoline(name: str = "") -> None:
    print(name, end="")
    print_now()
    loop.call_later(0.5, trampoline, name)


def hog():
    sum = 0
    for i in range(10_000):
        for j in range(10_000):
            sum += j
    return sum


loop = asyncio.get_event_loop()

""" loop.call_soon(trampoline)
loop.call_later(8, loop.stop)
loop.run_forever() """

""" # Sequential trampolines
loop.call_soon(trampoline, "First")
loop.call_soon(trampoline, "Second")
loop.call_soon(trampoline, "Third")

loop.call_later(8, loop.stop)

loop.run_forever() """

# Ref: [debug section](https://youtu.be/E7Yn5biBZ58?t=1625)
loop.set_debug(True)

loop.call_soon(trampoline, "First")
loop.call_soon(trampoline, "Second")
loop.call_soon(trampoline, "Third")

loop.call_later(15, hog)
loop.call_later(20, loop.stop)
loop.run_forever()

# generates the message:
# ``Executing <TimerHandle when=110222.109 hog() at <stdin>:1 created at <stdin>:1> took 9.391 seconds``
