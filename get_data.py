import subprocess
import time
import csv
import psutil

def safe_monitor_step(p, algo_name, run_id, thread_count, writer):
    try:
        with p.oneshot():
            cpu = p.cpu_percent(interval=None)
            mem = p.memory_info().rss / (1024 * 1024)
            threads = p.num_threads()
            ctx_switches = p.num_ctx_switches().voluntary + p.num_ctx_switches().involuntary
            uptime = time.time() - p.create_time()
            writer.writerow({
                "run_id": run_id,
                "algorithm": algo_name,
                "threads": thread_count,
                "pid": p.pid,
                "cpu_percent": round(cpu, 2),
                "memory_mb": round(mem, 2),
                "num_threads": threads,
                "context_switches": ctx_switches,
                "uptime": round(uptime, 2),
                "timestamp": round(time.time(), 2)
            })
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
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

        run_id = 0
        for i in range(10):
            for threads in range(1, 5):
                run_id += 1
                print(f"[Run {run_id}] Threads: {threads} | Launching bubble + merge concurrently")

                # Start both processes
                bubble_proc = subprocess.Popen(f"./p1_exec 1 {threads} bubble", shell=True)
                merge_proc = subprocess.Popen(f"./p1_exec 1 {threads} merge", shell=True)

                # Wrap in psutil
                bubble_ps = psutil.Process(bubble_proc.pid)
                merge_ps = psutil.Process(merge_proc.pid)

                # Monitor while either is alive
                while bubble_proc.poll() is None or merge_proc.poll() is None:
                    if bubble_proc.poll() is None:
                        safe_monitor_step(bubble_ps, "bubble", run_id, threads, writer)
                    if merge_proc.poll() is None:
                        safe_monitor_step(merge_ps, "merge", run_id, threads, writer)
                    time.sleep(0.5)

                print(f"[Run {run_id}] Finished monitoring both processes.")

    subprocess.run("make clean", shell=True)
    print("✅ Finished logging to process_metrics.csv")

if __name__ == "__main__":
    main()
    subprocess.run("make clean", shell=True)
    print("✅ Finished logging to process_metrics.csv")


