
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from opts import get_img_opts, get_viz_options
from transient_read import read_transient, get_transient_exr

def single_comp():
    """ Compare a single exr image"""
    opts = get_img_opts()
    file_name1 = os.path.join(opts.input_path1, f"{opts.input_name1}.exr")
    file_name2 = os.path.join(opts.input_path2, f"{opts.input_name2}.exr")
    img1 = get_transient_exr(file_name1)
    img2 = get_transient_exr(file_name2)
    img1 /= np.quantile(img1, opts.qnt)
    img2 /= np.quantile(img2, opts.qnt)
    # mean1 = img1.mean()
    # mean2 = img2.mean()
    # img2 *= mean1 / mean2
    # print(mean1, mean2)
    err = (img1 - img2).mean(axis = -1)
    err /= np.quantile(err, opts.qnt)
    # plt.imshow(img2 / np.quantile(img2, opts.qnt))
    plt.imshow(err.clip(-1, 1))
    plt.colorbar()
    plt.show()

def compositional_comp():
    """ Compare folder to folder (compositional) to see if the two methods converge to the same result """
    opts = get_viz_options()
    all_trans1 = read_transient(opts.input_path1, opts.input_name1, opts.num_transient)
    all_trans2 = read_transient(opts.input_path2, opts.input_name2, opts.num_transient)
    result1 = all_trans1.mean(axis = 0)
    result2 = all_trans2.mean(axis = 0)
    mean_1 = result1[result1 < np.quantile(result1, 0.9995)].mean()
    mean_2 = result2[result2 < np.quantile(result2, 0.9995)].mean()
    mse = ((result1 - result2) ** 2).mean(axis = -1)
    mse[mse > np.quantile(mse, 0.995)] = 0                 # remove huge values
    # err = np.log1p(mse)
    plt.imshow(mse)
    plt.title(f"Mean value: {mean_1:.5f}, {mean_2:.5f}")
    plt.colorbar()
    plt.savefig("./err-output.png")
    
def histogram_comp():
    """ Since statistically compare two method somehow leads to a huge gap between methods 
        So here is another comparison method to visualize the histogram of two methods 
    """
    num_bins = 100
    opts = get_viz_options()
    all_trans1 = read_transient(opts.input_path1, opts.input_name1, opts.num_transient)
    all_trans2 = read_transient(opts.input_path2, opts.input_name2, opts.num_transient)
    result1 = all_trans1.mean(axis = 0).mean(axis = -1)
    result2 = all_trans2.mean(axis = 0).mean(axis = -1)
    result1 = np.sqrt(result1)
    result2 = np.sqrt(result2)
    global_mini = min(result1.min(), result2.min()) * 0.9
    global_maxi = max(result1.max(), result2.max()) * 1.1
    sns.histplot(result1.ravel(), binrange = (global_mini, global_maxi), bins = num_bins, alpha = 1.0, log_scale = (False, True), kde = False, label = 'original sampling')
    sns.histplot(result2.ravel(), binrange = (global_mini, global_maxi), bins = num_bins, alpha = 0.4, log_scale = (False, True), kde = False, label = 'DARTS sampling')
    plt.legend()
    plt.grid(axis = 'both')
    plt.savefig("./sqrt-log-hist-.png", dpi=300)

if __name__ == "__main__":
    # compositional_comp()
    histogram_comp()
