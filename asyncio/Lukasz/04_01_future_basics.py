import asyncio

fut = asyncio.Future()
print(f"\nFuture has done is: {fut.done()}") # prints False as future has not done anything

print(f"\nFuture has cancelled is: {fut.cancelled()}") # prints False as nothing has been cancelled

# print(fut.result()) # raises an exception as future is not set yet

fut.set_result("result is set!")
print(f"\nHas result been set?: {fut.result()}")

print(f"\nHas future been done now: {fut.done()}")

print(f"\nWas the future cancelled: {fut.cancelled()}\n")
