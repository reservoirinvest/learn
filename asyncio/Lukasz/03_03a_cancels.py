import asyncio
import datetime

def print_now():
    print(datetime.datetime.now())

async def keep_printing(name: str= "") ->None:
    while True:
        print(name, end=" ")
        print_now()
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            print(name, " was cancelled")
            break

async def async_main() -> None:
    try:
        await asyncio.wait_for(
            asyncio.gather(
                keep_printing("First"),
                keep_printing("Second"),
                keep_printing("Third"),
            ),
            3
        )
    except asyncio.TimeoutError:
        print("oops, time's up!")

asyncio.run(async_main())
