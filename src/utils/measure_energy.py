import time
import torch
import subprocess
import pyRAPL

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

import time
import psutil

def measure_cpu_energy(model_fn, *args, **kwargs):
    # CPU-Leistungsaufnahme pro Kern in Watt schätzen (z.B. TDP / Anzahl Kerne)
    tdp_total_watt = 65  # Gesamt-TDP deiner CPU
    num_cores = psutil.cpu_count(logical=False)
    watt_per_core = tdp_total_watt / num_cores

    start_time = time.time()

    # Starte Messung der CPU-Auslastung
    cpu_percent_before = psutil.cpu_percent(interval=None)

    output = model_fn(*args, **kwargs)

    elapsed = time.time() - start_time

    # Durchschnittliche CPU-Auslastung während der Ausführung
    cpu_percent_after = psutil.cpu_percent(interval=None)
    avg_cpu_load = (cpu_percent_before + cpu_percent_after) / 2 / 100  # in [0,1]

    # Energie = Leistung * Zeit
    avg_power_watt = watt_per_core * num_cores * avg_cpu_load
    energy_joule = avg_power_watt * elapsed

    return output, elapsed, avg_power_watt, energy_joule
