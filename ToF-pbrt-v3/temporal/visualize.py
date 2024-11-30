import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sum_normalize(data: np.ndarray):
    return data / data.sum()

def exponential(x, sigma_t, scaler):
    return np.exp(-sigma_t * x) * scaler

def linear(x, a, b):
    return a * x + b


SIMGA_T = 20
SIGMA_A = 0.1

SOURCE_POS = np.float32([0.2779, 0.33, 0.30])

if __name__ == "__main__":
    debug_file = "../build/viz_output-neg.json"

    with open(debug_file, 'r') as file:
        json_data = json.load(file)
        output = json_data["values"]
    
    total_len = len(output) >> 1
    print(f"Total data item number: {len(output) >> 1}")
    value = input(f"Input the visualization id in [0, {total_len}): ")
    if not value.isdigit():
        raise ValueError("Not a valid number")
    value = int(value)
    if value < 0 or value >= total_len:
        raise ValueError(f"Not in range [0, {total_len})")
    base = value << 1
    da_vals = np.float32(output[base][:-1])
    tr_das = np.float32(output[base + 1][:-8])
    ray_o = np.float32(output[base + 1][-6:-3])
    ray_d = np.float32(output[base + 1][-3:])
    interval = output[base + 1][-8]
    remaining_time = output[base + 1][-7]

    ray2source = SOURCE_POS - ray_o
    dist2source = np.linalg.norm(ray2source)
    ray2source /= dist2source
    ray_d /= np.linalg.norm(ray_d)
    dot_prod = np.dot(ray_d, ray2source)
    print(f"Ray direction dot prod: {dot_prod}, ray distance: {dist2source}")

    plt.figure(1)
    plt.violinplot([da_vals, tr_das], showmeans = True)
    plt.title(f"Remaining time: {remaining_time:.5f}\nPoint: {ray_o}. Interval: {interval:.5f}")
    plt.xticks([1, 2], ['DA', 'DA * Tr'])
    plt.grid(axis = 'both')
    da_vals = sum_normalize(da_vals)
    tr_das = sum_normalize(tr_das)

    xs = np.arange(da_vals.shape[0], dtype = np.float32) * interval

    _, pcov = curve_fit(linear, xs, tr_das)
    print("Max covariance component for linear fitting:", pcov.max())

    _, pcov = curve_fit(exponential, xs, tr_das)
    print("Max covariance component for exponential fitting:", pcov.max())
    plt.figure(2)
    plt.scatter(xs, da_vals, c = 'r', s = 5)
    plt.plot(xs, da_vals, c = 'r', label = 'diffusion approximation')
    plt.scatter(xs, tr_das, c = 'b', s = 5)
    plt.plot(xs, tr_das, c = 'b', label = 'tr * da')

    exp_curve = sum_normalize(np.exp(-xs * (SIGMA_A + SIMGA_T)))
    plt.scatter(xs, exp_curve, c = 'g', s = 5)
    plt.plot(xs, exp_curve, c = 'g', label = 'exponential')
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()

