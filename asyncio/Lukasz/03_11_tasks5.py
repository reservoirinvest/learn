# Complete code for simple web crawler with graceful cancellation with wait
#

import asyncio
import sys
import time
from typing import Callable, Coroutine

import httpx

# added to prevent Event Loop exception...
if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

addr = 'http://landa.pl/crawl'


async def crawl2(prefix: str, url: str = "", ) -> None:
    url = url or prefix
    client = httpx.AsyncClient()

    try:
        res = await client.get(url)
    finally:
        await client.aclose()

    for line in res.text.splitlines():
        if line.startswith(prefix):
            task = asyncio.create_task(crawl2(prefix, line), name=line,)
            todo.add(task)


async def progress(url: str, algo: Callable[..., Coroutine],) -> None:

    task = asyncio.create_task(algo(url), name=url,)

    todo.add(task)
    start = time.time()

    while len(todo):
        done, _ = await asyncio.wait(todo, timeout=0.5)

        todo.difference_update(done)

        urls = (t.get_name() for t in todo)

        print(f"{len(todo)}: " + " ".join(sorted(urls))[-75:])

    end = time.time()
    print(f"Took {int(end-start)} seconds")


async def async_main() -> None:
    try:
        await progress(addr, crawl2)
    except asyncio.CancelledError:
        for task in todo:
            task.cancel()
        done, pending = await asyncio.wait(todo, timeout=1.0)
        todo.difference_update(done)
        todo.difference_update(pending)

        if todo:
            print("warning: new tasks were added while we were cancelling")


todo = set()
loop = asyncio.get_event_loop()
task = loop.create_task(async_main())
loop.call_later(10, task.cancel)
loop.run_until_complete(task)
