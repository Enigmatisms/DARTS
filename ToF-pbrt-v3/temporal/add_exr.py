import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from opts import get_viz_options
from transient_read import read_transient, get_processed_curves, colors


def add_two_exr():
    opts = get_viz_options()
    
    all_trans1 = read_transient(opts.input_path1, opts.input_name1, opts.num_transient)
    all_trans2 = read_transient(opts.input_path2, opts.input_name2, opts.num_transient)
    
    num_transient, h, w, _ = all_trans1.shape
    all_trans = all_trans1 + all_trans2
    qnt = np.quantile(all_trans, opts.qnt)
    print(f"Quantile: {qnt}")
    
    output_path = os.path.join(opts.input_path1, "../outputs")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in tqdm(range(num_transient)):
        output_name = os.path.join(output_path, f"{i:04d}.png")
        imageio.imwrite(output_name, ((all_trans[i, ...] / qnt).clip(0, 1) * 255).astype(np.uint8))
    print("The transient profile of two folders are added together.")
    
def composite():
    """ average the exr in a folder """
    opts = get_viz_options()
    all_trans1 = read_transient(opts.input_path1, opts.input_name1, opts.num_transient)
    result = all_trans1.mean(axis = 0)
    qnt = np.quantile(result, opts.qnt)
    output_path = os.path.join(opts.input_path1, "./outputs")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_name = os.path.join(output_path, f"composite-{opts.num_transient}.png")
    imageio.imwrite(output_name, ((result / qnt).clip(0, 1) * 255).astype(np.uint8))
    return result

if __name__ == "__main__":
    composite()