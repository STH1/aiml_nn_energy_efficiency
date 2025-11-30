import torch
from utils.measure_energy import measure_gpu_energy, measure_cpu_energy

def run_llm_inference(model, tokenizer, input_text="Hello", device=None, max_length=50):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    def forward_fn():
        with torch.no_grad():
            return model.generate(**inputs, max_length=max_length)

    if device.type == "cuda":
        return measure_gpu_energy(forward_fn)
    else:
        return measure_cpu_energy(forward_fn)
import torch
import concurrent.futures
import numpy as np

def run_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

def benchmark_llms(llm_dict, input_text="Hello", device=None, max_length=50, timeout_seconds=10):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    print(f"\n--- LLM inference on {device} ---")

    for name, (model, tokenizer) in llm_dict.items():
        output_data = run_with_timeout(
            run_llm_inference,
            timeout_seconds,
            model, tokenizer, input_text, device, max_length
        )

        if output_data is None:
            print(f"{name}: **TIMEOUT after {timeout_seconds}s** -> set NaN")
            results[name] = {
                "output": np.nan,
                "elapsed": np.nan,
                "avg_power": np.nan,
                "energy": np.nan
            }
            continue

        # Falls erfolgreich
        output, elapsed, avg_power, energy = output_data
        print(f"{name}: tokens={output.shape}, time={elapsed:.4f}s, power={avg_power:.2f} W, energy={energy:.2f} J")

        results[name] = {
            "output": output,
            "elapsed": elapsed,
            "avg_power": avg_power,
            "energy": energy
        }

    return results

