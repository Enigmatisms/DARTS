import sys
import imageio
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from pfm_helper import read_pfm, get_transient_exr

def outlier_rejection_mean(patch: np.ndarray, iter = 2, quantile = 0.995):
    """ Reject firefly induced mean shifting
    """
    mean = patch.mean()
    for _ in range(iter):
        diff = patch - mean
        diff_norm = np.linalg.norm(diff, axis = -1)
        qnt = np.quantile(diff_norm, quantile)
        mask = diff_norm <= qnt
        mean = patch[mask, :].mean()
    return mean

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ./histogram.py <path-to-hdrimage>")
        exit(1)
    path = sys.argv[1]
    if path.endswith('.exr'):
        image = get_transient_exr(path)
    else:
        image = read_pfm(path).astype(np.float32)
    image = image.mean(axis = -1)
    image = image[image > 1e-5]
    
    mean = image.mean()
    min_v, max_v = image.min(), image.max()
    qnt_99, qnt_80 = np.quantile(image, 0.99), np.quantile(image, 0.8)

    plt.title(f"{'exr' if path.endswith('exr') else 'pfm'} min/max = {min_v:.4f}/{max_v:.4f}\nmean={mean:.4f}, qnt99/qnt80={qnt_99:.4f}/{qnt_80:.4f}")
    plt.hist(image, bins = 500, range = (image.min(), image.max()))
    plt.grid(axis = 'both')
    plt.savefig(f"./hist-{'exr' if path.endswith('exr') else 'pfm'}.png", dpi = 200)
    