import subprocess
import time
import psutil
import torch
import joblib
import threading
import numpy as np
import pandas as pd
from torch import nn

# Define the model architecture
class RuntimePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load model and scaler
model = RuntimePredictor()
model.load_state_dict(torch.load("./model/runtime_predictor.pt"))
model.eval()
scaler = joblib.load("./model/scaler.save")

feature_names = ["threads", "cpu_percent", "memory_mb", "num_threads", "context_switches", "uptime"]

# Predict remaining runtime
def predict_runtime(proc: psutil.Process):
    try:
        with proc.oneshot():
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_info().rss / (1024 * 1024)
            threads = proc.num_threads()
            ctx_switches = proc.num_ctx_switches().voluntary + proc.num_ctx_switches().involuntary
            uptime = time.time() - proc.create_time()
            features = pd.DataFrame([[threads, cpu, mem, threads, ctx_switches, uptime]], columns=feature_names)
            scaled = scaler.transform(features)
            tensor = torch.tensor(scaled, dtype=torch.float32)
            with torch.no_grad():
                return model(tensor).item()
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return float('inf')

# Adjust nice levels using predictions
def adjust_priorities(children):
    predictions = []
    for proc in children:
        pred = predict_runtime(proc)
        predictions.append((proc, pred))
    predictions.sort(key=lambda x: x[1])
    try:
        base_nice = -15
        for i, (proc, _) in enumerate(predictions):
            try:
                proc.nice(base_nice + i)
            except psutil.AccessDenied:
                print(f"Could not set nice for PID {proc.pid}")
    except psutil.AccessDenied:
        print("⚠️ Need sudo to set negative nice values.")

# Monitor and log timing metrics
def monitor_and_log(processes, results, mode):
    arrival_times = {}
    start_times = {}
    end_times = {}

    ps_children = []
    for proc in processes:
        ps = psutil.Process(proc.pid)
        ps.cpu_percent(interval=None)
        time.sleep(0.1)
        children = ps.children(recursive=True)
        ps_children.append(children[0] if children else ps)
        arrival_times[ps_children[-1].pid] = time.time()

    # Track which PIDs are still running
    active_pids = set(p.pid for p in ps_children)

    while active_pids:
        for child in ps_children:
            try:
                if child.pid not in start_times and child.cpu_percent(interval=0.1) > 0:
                    start_times[child.pid] = time.time()

                if child.pid in active_pids and not child.is_running():
                    end_times[child.pid] = time.time()
                    active_pids.remove(child.pid)

            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                if child.pid in active_pids:
                    end_times[child.pid] = time.time()
                    active_pids.remove(child.pid)

        time.sleep(0.1)

    for child in ps_children:
        aid = arrival_times[child.pid]
        sid = start_times.get(child.pid, aid)
        eid = end_times.get(child.pid, time.time())
        results.append({
            "mode": mode,
            "pid": child.pid,
            "turnaround_time": round(eid - aid, 2),
            "waiting_time": round(sid - aid, 2)
        })

# Priority scheduling loop for multiple processes
def priority_scheduler_loop(processes):
    try:
        ps_children = []
        for proc in processes:
            ps = psutil.Process(proc.pid)
            ps.cpu_percent(interval=None)
            time.sleep(0.1)
            children = ps.children(recursive=True)
            ps_children.append(children[0] if children else ps)

        while any(proc.poll() is None for proc in processes):
            live_children = [ps for ps in ps_children if ps.is_running()]
            adjust_priorities(live_children)
            time.sleep(5)

    except Exception as e:
        print("Error in scheduling loop:", e)

# Run scheduling mode
def run_mode(name, use_nn, results):
    print(f"\n🚀 Running mode: {name}")
    subprocess.run("make clean && make", shell=True)

    algos = ["bubble", "merge"] * 5
    threads = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

    processes = []
    for i in range(10):
        cmd = f"./p1_exec 1 {threads[i]} {algos[i]}"
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    monitor_thread = threading.Thread(target=monitor_and_log, args=(processes, results, name))
    monitor_thread.start()

    if use_nn:
        sched_thread = threading.Thread(target=priority_scheduler_loop, args=(processes,))
        sched_thread.start()
        sched_thread.join()

    for p in processes:
        p.wait()

    monitor_thread.join()
    subprocess.run("make clean", shell=True)

# Main function
def main():
    results = []
    run_mode("Neural Scheduler", use_nn=True, results=results)
    run_mode("Default Linux Scheduler", use_nn=False, results=results)

    df = pd.DataFrame(results)
    df.to_csv("scheduling_results.csv", index=False)
    print("\n📊 Results saved to scheduling_results.csv:")
    print(df.groupby("mode")[["turnaround_time", "waiting_time"]].mean())

if __name__ == "__main__":
    main()

