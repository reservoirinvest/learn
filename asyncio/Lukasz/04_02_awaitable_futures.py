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

f = asyncio.Future()
loop = asyncio.get_event_loop()

""" loop.call_later(3, f.set_result, "this is my first result")

# print(f.result()) # Gives error as result is not set till after 3 seconds!
loop.run_until_complete(get_result(f))
print(f"\nResult is: {f.result()}") 

# Let us wrap multiple results
f = asyncio.Future()
loop.call_later(10, f.set_result, "another result")
res = loop.run_until_complete(get_result(get_result(get_result(f))))

print(f"\nThe result now is: {f.result()}") """

# Let us instead of result, handle exception
f = asyncio.Future()
loop.call_later(3, f.set_exception, ValueError("problem encountered!"))
res = loop.run_until_complete(get_result(f))
print(f"\nget_result returns: {res}\n")
# print(f"\nThe result now is: {f.result()}") # This will fail the program!