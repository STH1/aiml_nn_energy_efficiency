import torch
from utils.measure_energy import measure_gpu_energy, measure_cpu_energy

def run_mlp_inference(model, input_shape=(32,100), device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_shape, device=device)

    def forward_fn():
        with torch.no_grad():
            return model(x)

    if device.type == "cuda":
        return measure_gpu_energy(forward_fn)
    else:
        return measure_cpu_energy(forward_fn)

def benchmark_mlps(mlp_dict, input_shape=(32,100), device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    print(f"\n--- MLP inference on {device} ---")
    for name, model in mlp_dict.items():
        output, elapsed, avg_power, energy = run_mlp_inference(model, input_shape, device)
        print(f"{name}: time={elapsed:.4f}s, power={avg_power:.2f} W, energy={energy:.2f} J")
        results[name] = {"output": output, "elapsed": elapsed, "avg_power": avg_power, "energy": energy}
    return results
