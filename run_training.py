import threading
import subprocess
import filters

def run_thread(folder, shape):
    """
    run in a thread the training process for a folder and a shape
    """
    print("start trainig data for shape", shape)
    process = subprocess.Popen(
        ["python", "main.py", "-t", folder, shape],
        stdout=subprocess.PIPE,
        text=True
    )
    for line in process.stdout.readlines():
        print(line.strip())

if __name__ == "__main__":
    threads = []
    for shapes in filters.shapes:
        threads.append(threading.Thread(target = run_thread, args=("training_images/" + shapes['path'], shapes['name'],)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("training done!")

# run the following in a separate thread
#python main.py -t training_images/one_images/ "digit 1"
#python main.py -t training_images:two_images/ "digit 2"
#python main.py -t training_images/three_images/ "digit 3"
