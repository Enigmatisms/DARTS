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
    pfm_qnt = 0.99
    exr_qnt = 0.99
    if len(sys.argv) < 3:
        print("Usage: python ./hdr_comp.py <path-to-pfm> <path-to-exr> <flip first>")
        exit(1)
    do_flip = len(sys.argv) > 3
    if sys.argv[1].endswith('.exr'):
        print("The first input image is of .exr format. This is a special test case.")
        image_pfm = get_transient_exr(sys.argv[1])
    else:
        image_pfm = read_pfm(sys.argv[1]).astype(np.float64)
        if do_flip:
            image_pfm = np.flip(image_pfm, axis = 1)
    if sys.argv[2].endswith('.npy'):
        image_exr = np.load(sys.argv[2])
    else:
        image_exr = get_transient_exr(sys.argv[2])
    
    mean_pfm = outlier_rejection_mean(image_pfm, iter = 0)
    mean_exr = outlier_rejection_mean(image_exr, iter = 0)
    
    image_pfm *= mean_exr / mean_pfm
    qnt_pfm = np.quantile(image_pfm, pfm_qnt)
    qnt_exr = np.quantile(image_exr, exr_qnt)
    print(f"Quantile: ({qnt_pfm}, {qnt_exr}). Mean: ({mean_pfm, mean_exr})")
    print(image_pfm.shape, image_exr.shape)
    imageio.imwrite("pfm.png", ((image_pfm / qnt_pfm).clip(0, 1) * 255).astype(np.uint8))
    imageio.imwrite("exr.png", ((image_exr / qnt_exr).clip(0, 1) * 255).astype(np.uint8))
    
    diff_image = np.abs((image_pfm - image_exr).mean(axis = -1).clip(-1, 1))
    qnt_pfm_high = np.quantile(image_pfm, 0.997)
    qnt_pfm_low  = np.quantile(image_pfm, 0.003)
    mask = (image_pfm <= qnt_pfm_high) & (image_pfm >= qnt_pfm_low)
    image_pfm = image_pfm[mask]
    image_exr = image_exr[mask]
    print("Mean ABS diff:", ((image_pfm - image_exr) ** 2).mean())
    plt.imshow(diff_image)
    plt.colorbar()
    plt.savefig("./diff.png")
    