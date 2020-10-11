# Asyncio using tasks for primitive web-crawling
# ...using create_task()

# Ref: [EdgeDB](https://youtu.be/-CzqsgaXUM8?t=1780)

# create_task tells asyncio that now there is a coroutine that asyncio tracks.
# it will run it in the background of current execution context.
# ... i.e. it will run only when we are awaiting on something.

# await asyncio.sleep(0.5) is the place where background tasks are executed.

import asyncio
import time
from typing import Callable, Coroutine
import httpx

# Let us start by making a progress reporting async function.
addr = 'https://langa.pl/crawl'


# Note here that a callable returns a coroutine (static typing)
# First task is created here ... passing a coroutine and set a name.
#   ...setting name is a good practice for debugging
async def progress(
    url: str,
    algo: Callable[..., Coroutine],
) -> None:
    asyncio.create_task(
        algo(url),
        name=url,
    )

    # to really report progress we will need to know where we are.
    # we do it using the `todo` set.
    todo.add(url)
    start = time.time()

    # while there is stuff to do, let us print the status
    while len(todo):
        # formats the print, shows the length and some members (last 38)
        print(
            f"{len(todo)}: "
            + ", ".join(
                sorted(todo)
            )[-38:]
        )

        await asyncio.sleep(0.5)  # background tasks are executed here

    end = time.time()
    print(f"Took {int(end-start)}"
          + " seconds")

todo = set()  # create a todo set


# recreate the crawl set
async def crawl1(
    prefix: str, url: str = ""
) -> None:

    url = url or prefix
    client = httpx.AsyncClient()
    try:
        res = await client.get(url)
    finally:
        await client.aclose()
    for line in res.text.splitlines():
        if line.startswith(prefix):
            todo.add(line)
            await crawl1(prefix, line)
    todo.discard(url)

asyncio.run(progress(addr, crawl1))
