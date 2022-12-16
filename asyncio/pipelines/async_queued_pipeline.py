# asynch queued pipeline [dewaldabrie](https://gist.github.com/dewaldabrie/0d63881e9d0341762092745518fcc622)
import asyncio
import time

tasks = []

async def func(input_q, target_qs, task_id, param):
    print(f"{task_id}: Initialised with param: {param}")

    async def func_inner(input_q, target_qs, inpt):
        print(f"{task_id}: Recieved input: {inpt}")
        await asyncio.sleep(1)  # simulated IO delay
        outp = inpt
        for target_q in target_qs or []:
            print(f"{task_id}: T1 sending {outp}")
            await target_q.put(outp)
        input_q.task_done()

    while True:
        inpt = await input_q.get()
        print(f'{task_id}: Creating task with {task_id}_inner, input {inpt}.')
        tasks.append(asyncio.create_task(func_inner(input_q, target_qs, inpt)))


async def main():
    q2 = asyncio.Queue()
    coro2 = func(q2, [], 'T2', "hello T2")
    asyncio.create_task(coro2)

    q1 = asyncio.Queue()
    coro1 = func(q1, [q2], 'T1', "hello T1")
    asyncio.create_task(coro1)

    start_time = time.time()
    await q1.put(1)
    await q1.put(2)
    await q1.put(3)

    await q1.join()
    await q2.join()
    await asyncio.gather(*tasks)
    print(f"Duration: {time.time() - start_time}")

asyncio.run(main())