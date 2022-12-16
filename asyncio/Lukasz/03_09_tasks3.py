# Asyncio task for primitive crawling with concurrency
# ... using create_task

# Ref: [EdgeDB](https://youtu.be/-CzqsgaXUM8?t=2067)

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

# Let us recreate the function which is the same, except for one important difference:
# .... schedule crawls as background tasks, instead of awaiting for the crawl
# .... this is done using create_task, with a name given to it.


async def crawl2(
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

            # use create_task instead of await
            asyncio.create_task(
                crawl2(prefix, line),
                name=line,
            )
            todo.add(line)
    todo.discard(url)


todo = set()  # create a todo set
asyncio.run(progress(addr, crawl2))
