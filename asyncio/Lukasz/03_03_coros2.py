# Difference between coroutines and awaitables
# Ref: [EdgeDB youtube](https://youtu.be/-CzqsgaXUM8?t=883)

# The key things to note:
'''
1) An async function is a function that creates coroutines when called.
    It is defined using async def and can have await expressions inside it.

2) A coroutine is an object that is created by calling an async function.
    That object is awaited.
 
'''


import asyncio
import datetime
import inspect


def print_now():
    print(datetime.datetime.now())


async def keep_printing(name: str = "") -> None:
    while True:
        print(name, end=" ")
        print_now()
        await asyncio.sleep(0.5)


async def async_function() -> None:
    
    # keep_printing is an async function that is immediately awaited after it is called. 
    # It becomes a coroutine when it is called! The coroutine is awaitable.
    # awaiting on it is equivalent to calling it.
    
    await keep_printing()  

coroutine = async_function()
print(type(async_function))
print('\n')

print(type(coroutine))
print('\n')

print(inspect.isawaitable(async_function))
print('\n')

print(inspect.isawaitable(coroutine))
