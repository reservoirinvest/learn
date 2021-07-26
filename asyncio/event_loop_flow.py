# A brilliant explanation from [SO:](https://stackoverflow.com/a/59780868/7978112)

import asyncio
import types

def task_print(s):
    print(f"{asyncio.current_task().get_name()}: {s}")

async def other_task(s):
    task_print(s)

class AwaitableCls:
    def __await__(self):
        task_print("    'Jumped straight into' another `await`; "+ 
                    "the act of `await awaitable` *itself* doesn't 'pause' anything")
        yield
        task_print("    We're back to our awaitable object because that other task completed")
        asyncio.create_task(other_task("The event loop gets control when `yield` points "+
                        "(from an iterable coroutine) propagate up to the `current_task`" + 
                        "through a suitable chain of `await` or `yield from` statements"))

async def coro():
    task_print("  'Jumped straight into' coro; the `await` keyword itself "+
                    "does nothing to 'pause' the current_task")
    await AwaitableCls()
    task_print("  'Jumped straight back into' coro; we have another pending task, "+
                    "but leaving an `__await__` doesn't 'pause' the task any more than "+
                    "entering the `__await__` does")

@types.coroutine
def iterable_coro(context):
    task_print(f"`{context} iterable_coro`: pre-yield")
    yield None # None or a Future object are the only legitimate yields to the task in asyncio
    task_print(f"`{context} iterable_coro`: post-yield")

async def original_task():
    asyncio.create_task(other_task("Aha, but a (suitably unconsumed) *`yield`* "+
            "DOES 'pause' the current_task allowing the event scheduler to `_wakeup` another task"))

    task_print("Original task")
    await coro()
    task_print("'Jumped straight out of' coro. Leaving a coro, as with leaving/entering any awaitable,"+
                " doesn't give control to the event loop")
    res = await iterable_coro("await")
    assert res is None
    asyncio.create_task(other_task("This doesn't run until the very end because the generated None "+
                "following the creation of this task is consumed by the `for` loop"))
    for y in iterable_coro("for y in"):
        task_print(f"But 'ordinary' `yield` points (those which are consumed by the `current_task` itself)"+
                f" behave as ordinary without relinquishing control at the async/task-level; `y={y}`")
    task_print("Done with original task")

asyncio.get_event_loop().run_until_complete(original_task())