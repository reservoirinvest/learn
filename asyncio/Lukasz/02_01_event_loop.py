import asyncio

loop = asyncio.get_event_loop()

# loop.run_forever() # runs for ever!!

# loop.run_until_complete(asyncio.sleep(5)) # sleeps for 5 seconds

import datetime
def print_now():
    print(datetime.datetime.now())
    
loop.call_soon(print_now) # print_now is without parenthesis(). It is a registered callback.
loop.call_soon(print_now)

loop.run_until_complete(asyncio.sleep(5))