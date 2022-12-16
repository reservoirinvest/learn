# Example to limit concurrency in asyncio
# Ref: (SO: 48483348)[https://stackoverflow.com/a/48483348/7978112]

import asyncio
import random

async def download(code):
    wait_time = random.randint(1, 3)
    print(f"downloading {code} will take {wait_time} seconds")
    await asyncio.sleep(wait_time)
    print(f"downloaded {code}")

async def main(loop):
    no_concurrent=3
    dltasks = set()
    i=0

    while i < 9:
        if len(dltasks) >= no_concurrent:
            # wait for some download to finish before adding a new one
            _done, dltasks = await asyncio.wait(dltasks, return_when=asyncio.FIRST_COMPLETED)
        dltasks.add(loop.create_task(download(i)))
        i+=1
    
    # Wait for the remaining downloads to finish
    await asyncio.wait(dltasks)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(loop))
    finally:
        loop.close()