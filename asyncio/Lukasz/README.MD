# Intro

This is a series of programs from Łukasz Langa's tutorials on [EdgeDB](https://youtu.be/Xbl7XjFYsN4)

# Critical concepts
Asyncio has the following parts / concepts:

1. `async def` function - which can have *await* statements in it.
   
2. asyncio `callables` - which can be:
   * *async def* functions, or,
   * any other 3rd party function that supports asynchronous call-backs
  
3. `coroutine` - is an object that is created by calling an *async def* function. This couroutine - called object - is awaited.
   
4. `awaitable` - any object that can be used in an await expression is called an awaitable. *aw*s- are:
   * *async def*, 
   * *await* statements, 
   * *coroutine* objects, or, 
   * *future* objects 
   
   ...that is ready to be scheduled in an event loop

5. `event` loop - the loop in a thread that schedules and runs the awaitables
   
6. `gather` - which runs *aw* in the aws sequence concurrently. 
   * If any aw is a coroutine, it is automatically scheduled as a *Task*. Upon successful completion the result is an aggregate list of values in the order of aws. 
   * If *return_exception* is `False`, the first raised exception is immediately propogated to the task that awaits on gather(). 
     * However, other aws are *not cancelled*. 
   * If *return_exception* is `True`, exeptions are treated the same as successful results and aggregated in the list. 
   * If gather() is cancelled, all submitted pending aws are also cancelled. 
   * If any Task or Future in aws is cancelled, the gather() call is not cancelled, to prevent cancellation of one submitted Task to cause cancellation of others.
  
7. `Task` - is used to schedule the coroutines concurrently in the event loop. 
   * To cancel a *Task* use `cancel()` method. 
     * It will cause Task to throw a `CancelledError` exception in the wrapped coroutine.
  
8. `wait` - an asyncio method that waits on a coroutine. 
   * It blocks till `return_when` and returns two set of tasks - *done* and *pending*.
   * *return_when* can be [FIRST_COMPLETED | FIRST_EXCEPTION | ALL_COMPLETED].
  
9.  `wait_for` - an asyncio method that waits on an aw with a timeout. 
    * If the aw is a coroutine it automatically schedules as a task. It generates TimeoutError exception after timeout and cancels the task.
  
10. `asyncio.run` - executes the *coroutine* and returns the result. 
    * It runs the passed *coroutine*, manages the asyncio *event* loop and finalizes async generators. 
    * It cannot be called if another async event loop is running in the same thread. 
    * If `debug` is true, it runs in debug mode.
    * *asyncio.run* should be used as the main entry for asyncio programs, should ideally be called only once - as it creates a new event loop and closes it at the end.
  
11. `asyncio.create_task` - wraps *coroutine* into a *Task* and schedules its execution. 
    * It returns a *Task* object. 
    * Its name can be set using `Task.set_name()` method. 
    * The *Task* is exceuted in the loop returned by `get_running_loop()`. 
    * If there is no running loop, a `RuntimeError` is generated.
