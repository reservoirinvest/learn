# REFERENCE program for concurrency

import asyncio
import random
from pprint import pprint
from typing import Iterable
from time import perf_counter


async def a_task(t: int, # task no 
                 w: float,  # random wait in seconds
                 threshold: float=0.4, # to generate exception
                 ) -> float: 
    await asyncio.sleep(1)
    if w > threshold:
        raise Exception(f"{w} of Task-{t} is too high")
    return w

async def do_tasks(it: Iterable, timeout: float, threshold: float):

    i = 0
    tasks = set()
    results = set()

    L = 1 # Loop number

    while i < total:

        b = 0
        
        while b < concurrent:

            if i != total: # prevents fragments within concurrent

                tasks.add(asyncio.create_task(a_task(i, it[i], threshold=threshold), 
                        name=f"Task-{i}"))

                i +=1
            
            b += 1

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        print(f"\nTasks in loop {L} just after asyncio.wait")

        for t in tasks:
            print(t)
        print("-------------\n")
        print(f"\nDone just after asyncio.wait of loop {L}")

        for d in done:
            print(d)
        print("-------------\n")
        print(f"\nPending just after asyncio.wait of loop {L}")

        if pending:
            for p in pending:
                print(p)
            print("-------------\n")
        else:
            print("! Nothing is pending !")
            print("-------------\n")

        tasks.difference_update(done) # removes the completed tasks

        results.update(done)

        for t in done:
            t.cancel()

        if pending:
            print(f'Alert! {[p.get_name() for p in pending]} pending tasks will be killed in 2 seconds')
            d, p = await asyncio.wait(pending, timeout=2)
            results.update(d)
            [c.cancel for c in d | p]
            tasks.difference_update(d|p)

        L +=1

    return results

random.seed(444)
timeout = 0.5 
threshold = 0.4 # task exception threshold

total = 10 # total number of items to be processed
it = [random.random() for _ in range(total)]

concurrent = 3 # number of concurrent items to be processed

start = perf_counter()
output = asyncio.run(do_tasks(it, timeout, threshold))
elapsed = perf_counter()

print("Here are the results:")

for o in output:
    print(o)

print(f"\nDone do_tasks({len(output)}) in {elapsed:0.2f} secs\n")