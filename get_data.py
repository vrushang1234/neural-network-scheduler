import subprocess
import time
import csv
import psutil
import threading

def safe_monitor_step(p, algo_name, run_id, thread_count, writer, lock):
    try:
        with p.oneshot():
            cpu = p.cpu_percent(interval=None)
            mem = p.memory_info().rss / (1024 * 1024)
            threads = p.num_threads()
            ctx_switches = p.num_ctx_switches().voluntary + p.num_ctx_switches().involuntary
            uptime = time.time() - p.create_time()
            with lock:
                writer.writerow({
                    "run_id": run_id,
                    "algorithm": algo_name,
                    "threads": thread_count,
                    "pid": p.pid,
                    "cpu_percent": cpu,
                    "memory_mb": round(mem, 2),
                    "num_threads": threads,
                    "context_switches": ctx_switches,
                    "uptime": round(uptime, 2),
                    "timestamp": round(time.time(), 2)
                })
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        pass

def monitor_children(proc, algo_name, run_id, thread_count, writer, lock):
    try:
        root_ps = psutil.Process(proc.pid)
        root_ps.cpu_percent(interval=None)
        monitored = []

        while proc.poll() is None:
            time.sleep(0.1)
            children = root_ps.children(recursive=True)

            for child in children:
                if child.pid not in [p.pid for p in monitored]:
                    child.cpu_percent(interval=None)
                    monitored.append(child)

            for p in monitored:
                safe_monitor_step(p, algo_name, run_id, thread_count, writer, lock)
    except psutil.NoSuchProcess:
        pass

def main():
    subprocess.run("make clean && make", shell=True, check=True)

    with open("process_metrics.csv", "w", newline="") as f:
        fieldnames = [
            "run_id", "algorithm", "threads", "pid",
            "cpu_percent", "memory_mb", "num_threads",
            "context_switches", "uptime", "timestamp"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        lock = threading.Lock()

        run_id = 0
        for i in range(1):
            for threads in range(1, 6):
                run_id += 1
                print(f"[Run {run_id}] Threads: {threads} | Launching bubble + merge concurrently")

                bubble_proc = subprocess.Popen(f"./p1_exec 1 {threads} bubble", shell=True)
                merge_proc = subprocess.Popen(f"./p1_exec 1 {threads} merge", shell=True)

                bubble_thread = threading.Thread(target=monitor_children,
                                                 args=(bubble_proc, "bubble", run_id, threads, writer, lock))
                merge_thread = threading.Thread(target=monitor_children,
                                                args=(merge_proc, "merge", run_id, threads, writer, lock))

                bubble_thread.start()
                merge_thread.start()

                bubble_thread.join()
                merge_thread.join()

                print(f"[Run {run_id}] Finished monitoring both processes.")

    subprocess.run("make clean", shell=True)
    print("✅ Finished logging to process_metrics.csv")

if __name__ == "__main__":
    main()

