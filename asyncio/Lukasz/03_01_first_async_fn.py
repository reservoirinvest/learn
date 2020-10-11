import asyncio
import datetime


def print_now():
    print(datetime.datetime.now())


async def keep_printing(name: str = "") -> None:
    while True:
        print(name, end=" ")
        print_now()
        await asyncio.sleep(0.5)

# asyncio.run(keep_printing("first")) # continuously runs forever

""" # waits for a timeout and creates a TimeoutError exception
asyncio.run(asyncio.wait_for(keep_printing(), 3)) """


# Let us use a main function to get into the asyncio world.
# This is a graceful way of handling asyncio functions
async def async_main() -> None:
    try:
        await asyncio.wait_for(keep_printing('Hey'), 3)
    except asyncio.TimeoutError:
        print("I have timed out!!!")

asyncio.run(async_main())
