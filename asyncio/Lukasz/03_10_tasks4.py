# Using tasks on create_task for better control of errors
# Ref: [EdgeDB](https://youtu.be/-CzqsgaXUM8?t=2279)


import asyncio
import sys
import time
from typing import Callable, Coroutine

import httpx

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Let us start by making a progress reporting async function.
addr = 'https://langa.pl/crawl'


async def progress(
    url: str,
    algo: Callable[..., Coroutine],
) -> None:

    # we will store the create_task into a variable
    task = asyncio.create_task(
        algo(url),
        name=url,
    )

    todo.add(task)  # we will add task instead of line
    start = time.time()

    while len(todo):

        # we will use asyncio.wait function that takes and collection of tasks
        # ... our todo set is great for this
        # ... and waits for them to complete
        # ... unlike wait_for, asyncio.wait will not raise an exception
        # ... it instead gives us two tasks, done and the ones which are pending

        done, _pending = await asyncio.wait(todo, timeout=0.5)

        # to clean up our todo set, we are removing the done task from it
        todo.difference_update(done)

        # we will report progress as things are going on
        urls = (t.get_name() for t in todo)

        # prints the current status
        print(f"{len(todo)}: " + " ".join(sorted(urls))[-75:])

    end = time.time()
    print(f"Took {int(end-start)} seconds")


async def crawl3(
    prefix: str, url: str = "",
) -> None:
    url = url or prefix
    client = httpx.AsyncClient()
    try:
        res = await client.get(url)
    finally:
        await client.aclose()

    for line in res.text.splitlines():
        if line.startswith(prefix):
            task = asyncio.create_task(
                crawl3(prefix, line),
                name=line,
            )

            todo.add(task)

todo = set()
asyncio.run(progress(addr, crawl3))
