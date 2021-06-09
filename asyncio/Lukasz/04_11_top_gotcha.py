import asyncio

def indent(count: int) -> str:
    return "  " * (6-(count*3))

async def example(count: int) -> str:
    await asyncio.sleep(0)
    if count == 0:
        return "result"

    for i in range(count):
        await asyncio.sleep(i)
    return await example(count-1)

