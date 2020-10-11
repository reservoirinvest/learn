# simplest example of asyncio
# from https://stackoverflow.com/questions/36342899
# see response from Mikhail https://stackoverflow.com/a/36415477/7978112
# this with https://developer.ibm.com/tutorials/ba-on-demand-data-python-3/ 
# ... should help to understand the wrokings of asyncio well


import asyncio


async def doit(i):
    print(f"Start {i}")
    await asyncio.sleep(3)
    print(f"End {i}")
    return i

""" if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    # futures = [asyncio.ensure_future(doit(i), loop=loop) for i in range(10)]
    futures = [loop.create_task(doit(i)) for i in range(10)] # recommended way!
    # futures = [doit(i) for i in range(10)]
    result = loop.run_until_complete(asyncio.gather(*futures))
    print(result)
 """

async def msg(text):
    await asyncio.sleep(0.1)
    print(text)

async def long_operation():
    print('long_operation started')
    await asyncio.sleep(3)
    print('long_operation finished')

async def main():
    await msg('first')

    # Now you want to start long_operation, but you don't want to wait till it is finished
    # long_operation should be started, but second msg should be printed immediately.
    # Create task to do so:
    task = asyncio.ensure_future(long_operation())
    
    # await long_operation() # just to test the difference

    await msg('second')

    # Now, when you want, you can await task finised:
    await task

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
