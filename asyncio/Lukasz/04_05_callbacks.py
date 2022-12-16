import asyncio

loop = asyncio.get_event_loop()

def callback(fut: asyncio.Future) -> None:
    print("called by", fut)

f = asyncio.Future()
f.add_done_callback(callback)
f.add_done_callback(lambda f: loop.stop())
f.set_result("yup")
loop.run_forever()

# futures always invoke a loop / current loop to schedule.
# this adheres to 'fair' scheduling.