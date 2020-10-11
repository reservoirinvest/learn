# This program highlights tasks
# Please install httpx before running it...

# Coroutines are found in collections.abc alongside its parent, the awaitables. They are core features of python built into the interpreter.
# note that if you use twisted, tornado, curio, ... or any other framework, you will not use asyncio's futures and tasks.
# cororoutine is a low-level block which doesn't know about asyncio concept like event loop or cancellations.
# that is why asyncio functions are wrapped in `tasks`

# Problems with the code:

# 1. Backend activities also are reporting their status. Reporting progress from the same task is a no-no, if it is just a print or log.
# 2. Calling crawl0 within crawl0 is recursive. This may be an issue if there are deep websites. Avoid it.
# 3. `await crawl` is recreating a blocking environment, as we are only crawling on one particular url. Not using concurrency.
# 4. We are getting an async client just like that. We should use a `context manager` instead.

import asyncio

import httpx


async def crawl0(
    prefix: str, url: str = ""
) -> None:

    url = url or prefix
    print(f"Crawling {url}")

    client = httpx.AsyncClient()
    try:
        res = await client.get(url)
    finally:
        await client.aclose()

    for line in res.text.splitlines():
        if line.startswith(prefix):
            await crawl0(prefix, line)

asyncio.run(crawl0("https://langa.pl/crawl/"))
