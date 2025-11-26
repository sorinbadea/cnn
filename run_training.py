import asyncio
import filters

async def run_task(folder, shape):
    """
    run async training process for a folder and a shape
    """
    print("start training data for shape", shape)
    process = await asyncio.create_subprocess_exec(
        "python", "main.py", "-t", folder, shape,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    # Read output line by line asynchronously
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        print(line.strip())
    await process.wait()

async def show(stop_event):
    timeout = 0
    while not stop_event.is_set():
        print('elapsed time',timeout,'seconds',end='\r')
        await asyncio.sleep(1)
        timeout += 1
    print(f"elapsed {timeout} seconds")

async def main():
    stop_event = asyncio.Event()
    # Create the show task
    show_elapsed_task = asyncio.create_task(show(stop_event))
    tasks = []
    for shapes in filters.shapes:
        tasks.append(asyncio.create_task(
            run_task("training_images/" + shapes['path'], shapes['name']))
        )
    # Wait for all training tasks to complete
    await asyncio.gather(*tasks)
    # Now stop the timer
    stop_event.set()
    await show_elapsed_task
    print("training done!")

if __name__ == "__main__":
    asyncio.run(main())