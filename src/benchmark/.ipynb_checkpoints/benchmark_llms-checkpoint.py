import torch
from utils.measure_energy import measure_gpu_energy, measure_cpu_energy

def run_llm_inference(model, tokenizer, input_text="Hello", device=None, max_length=50):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    def forward_fn():
        with torch.no_grad():
            return model.generate(**inputs, max_length=max_length)

    if device.type == "cuda":
        return measure_gpu_energy(forward_fn)
    else:
        return measure_cpu_energy(forward_fn)

def benchmark_llms(llm_dict, input_text="Hello", device=None, max_length=50):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    print(f"\n--- LLM inference on {device} ---")
    for name, (model, tokenizer) in llm_dict.items():
        output, elapsed, avg_power, energy = run_llm_inference(model, tokenizer, input_text, device, max_length)
        print(f"{name}: tokens={output.shape}, time={elapsed:.4f}s, power={avg_power:.2f} W, energy={energy:.2f} J")
        results[name] = {"output": output, "elapsed": elapsed, "avg_power": avg_power, "energy": energy}
    return results
