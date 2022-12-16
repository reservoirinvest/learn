# A simple example using wait instead of gather to capture embedded exception error
# Note that the exception error is captured using [`BaseException` from python 3.8!](https://stackoverflow.com/a/67969473/7978112) 

import asyncio
import random
from time import perf_counter
from typing import Iterable
from pprint import pprint

async def coro(t, i, threshold=0.4):
    await asyncio.sleep(i)
    if i > threshold:
        # For illustration's sake - some coroutines may raise,
        # and we want to accomodate that and just test for exception
        # instances in the results of asyncio.gather(return_exceptions=True)
        raise Exception(f"{i} of Task-{t} is too high")
    return i

async def main(it: Iterable, timeout: float) -> tuple:
    tasks = [asyncio.create_task(coro(i+1, d), name=f"Task-{i+1}") for i, d in enumerate(it)]
    await asyncio.wait(tasks, timeout=timeout)
    return tasks  # *not* (done, pending)

timeout = 0.5
random.seed(444)
n = 10
it = [random.random() for _ in range(n)]
start = perf_counter()
tasks = asyncio.run(main(it=it, timeout=timeout))
elapsed = perf_counter() - start
print(f"Done main({n}) in {elapsed:0.2f} seconds\n")
pprint(tasks)
print('----')


# retrieve the tasks
res = []
for t in tasks:
    try:
        r = t.result()
    except BaseException as e:
        res.append(e)
    else:
        res.append(r)
pprint(res)