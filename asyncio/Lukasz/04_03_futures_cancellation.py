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
loop.call_later(2, f.set_result, "this is my result")
output = loop.run_until_complete(get_result(f))
print(output)

f = asyncio.Future()
loop.call_later(3, f.set_result, "another result")
output = loop.run_until_complete(get_result(f))
print(output)

f = asyncio.Future()
loop.call_later(1, f.set_exception, ValueError("problem encountered"))
output = loop.run_until_complete(get_result(get_result(get_result(f))))
print(f"\nget_result returns: {output}\n")

# cancellation exception propogates! 
# As they are base exceptions like keyboard interrupt
f = asyncio.Future()
loop.call_later(5, f.cancel)
output = loop.run_until_complete(get_result(f))
print(f"\nget_result upon cancellation returns: {output}\n")