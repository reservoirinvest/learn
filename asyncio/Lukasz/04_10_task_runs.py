import asyncio

# testing await / yield from chains
async def example(count: int) -> str:
    await asyncio.sleep(0)
    if count == 0:
        return "result"

    for i in range(count):
        await asyncio.sleep(i)

    return await example(count - 1)

class TraceStep(asyncio.tasks._PyTask):
    def _Task__step(self, exc=None):
        print(f"<step name={self.get_name()} done={self.done()}>")
        result = super()._Task__step(exc=exc)
        print(f"</step name={self.get_name()} done={self.done()}>")


loop = asyncio.get_event_loop()

loop.set_task_factory(lambda loop, coro: TraceStep(coro, loop=loop))

# loop.run_until_complete(example(5))

async def example1(count: int) -> str:

    print(f"  {count} Before the first await")
    await asyncio.sleep(0)
    print(f"  {count} After the first await")

    if count == 0:
        print(f"  {count} Returning result")
        return "result"

    for i in range(count):
        print(f"  {count} Before await inside loop iteration", i)
        await asyncio.sleep(i)
        print(f"  {count} After await inside loop iteration", i)

    print(f"     {count} Before await on example ({count - 1})", i)

    return await example1(count - 1)

loop.run_until_complete(example1(1))