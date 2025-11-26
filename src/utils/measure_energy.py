import time
import torch
import subprocess

def measure_gpu_energy(model_fn, *args, **kwargs):
    start_time = time.time()
    output = model_fn(*args, **kwargs)
    elapsed = time.time() - start_time

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        avg_power_watt = float(result.stdout.strip())
    except:
        avg_power_watt = 200

    energy_joule = avg_power_watt * elapsed
    return output, elapsed, avg_power_watt, energy_joule

def measure_cpu_energy(model_fn, *args, **kwargs):
    tdp_watt = 65

    start_time = time.time()
    output = model_fn(*args, **kwargs)
    elapsed = time.time() - start_time

    energy_joule = tdp_watt * elapsed
    avg_power_watt = tdp_watt
    return output, elapsed, avg_power_watt, energy_joule
