import os
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from opts import get_tran_comp_options
from transient_read import read_transient, get_processed_curves, colors

CONSOLE = Console(width = 128)

if __name__ == "__main__":
    opts = get_tran_comp_options()

    crop_info = None
    if opts.crop_rx > 0 and opts.crop_ry > 0:
        crop_info = (opts.crop_x, opts.crop_y, opts.crop_rx, opts.crop_ry)
        CONSOLE.log("Cropping:", crop_info)
        
    actual_time = 1
    for i, (folder, name, legend) in enumerate(zip(opts.folders, opts.input_names, opts.legends)):
        input_path = os.path.join(opts.input_dir, folder)
        trans = read_transient(input_path, name, opts.num_transient, crop_info, extension = opts.ext, identity = False)
        num_transient, h, w, _ = trans.shape

        CONSOLE.log(f"Curve '{legend}' loaded from: {input_path}")
        
        xs1, (curves_1,), actual_time = get_processed_curves(opts, trans, num_transient, opts.qnt)
        plt.scatter(xs1, curves_1, s = 5, c = colors[i])
        plt.plot(xs1, curves_1, label = f'{legend}', c = colors[i])
    
    plt.legend()
    plt.xlim((0, actual_time))
    plt.xlabel("Time (ns)")
    plt.grid(axis = 'both')
    plt.show()
