import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from opts import get_viz_options
from transient_read import get_transient_exr

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
    opt_normalize = False
    opts = get_viz_options()

    file_1 = os.path.join(opts.input_path1, f"{opts.input_name1}.exr")
    file_2 = os.path.join(opts.input_path2, f"{opts.input_name2}.exr")

    print(file_1)
    print(file_2)

    image1 = get_transient_exr(file_1).mean(axis = -1)
    mean_1 = image1.mean()
    qnt_1 = np.quantile(image1, opts.qnt)
    image1 /= qnt_1
    image2 = get_transient_exr(file_2).mean(axis = -1)
    mean_2 = image2.mean()
    qnt_2 = np.quantile(image2, opts.qnt)
    image2 /= qnt_2
    print(f"Quantiles: {qnt_1:.6f}, {qnt_2:.6f}")
    if opt_normalize:
        import cvxpy as cp
        x = cp.Variable(1)
        cost = cp.sum_squares(image1 - image2 * x)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve(verbose = False)
        image2 *= x.value
        print(f"Best image scaler cvx: {x.value}")
    else:
        scaler = outlier_rejection_mean(image1) / outlier_rejection_mean(image2)
        print(f"Best image scaler outlier-reject mean: {scaler}")
        image2 *= outlier_rejection_mean(image1) / outlier_rejection_mean(image2)
    neg = image1 < image2
    diff_image = image1 - image2
    print(np.abs(diff_image).mean())
    # diff_image = np.log(np.abs(image1 - image2).clip(0, 1) + 1)
    # diff_image[neg] *= -1
    # greater = np.abs(diff_image) > 0.1
    # diff_image[greater] = 1
    # diff_image[~greater] = 0
    
    num_bins = 500
    plt.subplot(2, 1, 1)
    plt.title(f"({1 if opt_normalize else 0}) {opts.input_name1} = {qnt_1:.5f}, {mean_1:.5f}\n{opts.input_name2} = {qnt_2:.5f}, {mean_2:.5f}")
    sns.histplot(image1.ravel(), binrange = (image2.min(), image2.max() * 0.1), bins = num_bins, alpha = 1.0, 
                log_scale = (False, True), kde = False, label = f'{opts.input_name1}')
    plt.legend()
    plt.grid(axis = 'both')
    plt.subplot(2, 1, 2)
    sns.histplot(image2.ravel(), binrange = (image2.min(), image2.max() * 0.1), bins = num_bins, alpha = 1.0, 
                log_scale = (False, True), kde = False, label = f'{opts.input_name2}')
    plt.legend()
    plt.grid(axis = 'both')
    plt.savefig("./hist-comp.png", dpi = 300)
    plt.clf()
    plt.cla()

    diff_image = np.clip(diff_image, -0.1, 0.1)
    plt.imshow(diff_image)
    plt.colorbar()
    plt.savefig("./output.png", dpi=300)