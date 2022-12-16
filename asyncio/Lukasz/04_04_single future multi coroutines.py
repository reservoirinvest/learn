import asyncio
from typing import Awaitable 

async def get_result(awaitable: Awaitable) -> str:
    try:
        result = await awaitable
    except Exception as e:
        print("\noops!", e)
        return "no result :("
    else:
        return result

loop = asyncio.get_event_loop()

f = asyncio.Future()
loop.call_later(5, f.set_result, "final result")
output = loop.run_until_complete(asyncio.gather(get_result(f), get_result(f), get_result(f)))
print(output)