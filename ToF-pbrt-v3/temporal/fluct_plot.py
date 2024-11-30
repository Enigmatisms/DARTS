"""
Plotting the fluctation
@date 2023.1.7
"""

import os
import sys
import json
sys.path.append("..")

import natsort
import numpy as np
import seaborn as sns
import concurrent.futures
import matplotlib.pyplot as plt
from rich.console import Console
from utils.plt_utils import COLORS 
from matplotlib import font_manager
from opts import get_tran_comp_options
from utils.post_process import enumerate_filtered_folders
from transient_read import read_transient, read_hdr_file
from scipy.interpolate import CubicSpline, interp1d

CONSOLE = Console(width = 128)
VERSION = "1.0"
MAX_WORKERS = 10

def process_frame(folders: list, name: str, ext: str, idx: int, crop_info:tuple = None):
    values = []
    for folder in folders:
        hdr_file = os.path.join(folder, f"{name}_{idx}.{ext}")
        if not os.path.exists(hdr_file):
            hdr_file = os.path.join(folder, f"{name}_{idx:04d}.{ext}")
            if not os.path.exists(hdr_file):
                CONSOLE.log(f"[red]Error: path '{folder}:{name}:{idx}:{ext}' does not exist[/red]")
                exit(1)
        hdr_img = read_hdr_file(hdr_file, ext)
        if crop_info is not None:
            if len(crop_info) > 4:
                sx, sy, ex, ey = crop_info[:-1]
                if ex == 0 or ey == 0:
                    return hdr_img
            else:
                sx = crop_info[0] - crop_info[2]
                ex = crop_info[0] + crop_info[2] + 1
                sy = crop_info[1] - crop_info[3]
                ey = crop_info[1] + crop_info[3] + 1
            hdr_img = hdr_img[sy:ey, sx:ex, :]
        values.append(hdr_img.mean())
    return (idx, np.float32(values))

def double_row_formatter(num_transient, start_time, length, sol, interval = 8):
    times = np.arange(num_transient, dtype = np.float32) * length / num_transient + start_time
    times /= sol
    labels   = []
    tick_pos = []
    for frame_id, time_point in enumerate(times):
        if frame_id % interval == 0:
            labels.append(f"{frame_id + 1}\n{time_point:.2f}")
            tick_pos.append(frame_id)
    tick_pos.append(num_transient - 1)
    labels.append(f"{num_transient}\n{(start_time + length) / sol:.2f}")
    return tick_pos, labels

def plot_smooth_error_curve(xs, gt_means, data, label, color, alpha = 0.2, fine_num = 0, nullify_point = 0):
    """ Draw smoothed error range """
    std_devs = np.std(data, axis=1)
    smoothed_lower = gt_means - std_devs
    smoothed_upper = gt_means + std_devs
    if fine_num > 1:
        # first we linear interpolate between points
        linear_xs = np.linspace(xs.min(), xs.max(), xs.shape[0] * 2 - 1)
        interp_func_lower = interp1d(xs, smoothed_lower, kind='linear', fill_value='extrapolate')
        interp_lower = interp_func_lower(linear_xs)
        interp_func_upper = interp1d(xs, smoothed_upper, kind='linear', fill_value='extrapolate')
        interp_upper = interp_func_upper(linear_xs)
        # Cubic spline interpolate
        spline_lower = CubicSpline(linear_xs, interp_lower)
        spline_upper = CubicSpline(linear_xs, interp_upper)
        
        def safe_interpolate(inputs, interp_func):
            vals = interp_func(inputs)
            vals[inputs < xs[nullify_point]] = 0
            return vals
            
        finer_xs = np.linspace(xs.min(), xs.max(), xs.shape[0] * fine_num)
        plt.fill_between(finer_xs, safe_interpolate(finer_xs, spline_lower), safe_interpolate(finer_xs, spline_upper), color=color, alpha = alpha, label=f'{label}')
    else:
        plt.fill_between(xs, gt_means - std_devs, gt_means + std_devs, color=color, alpha = alpha, label=f'{label}')

if __name__ == "__main__":
    parser = get_tran_comp_options(delayed_parse = True)
    parser.add_argument("--interp_num",    default = 0, help = "Spline interpolation make-finer number (default 0 --> no spline interpolation)", type = int)
    parser.add_argument("--nullify_point", default = 0, help = "Set the interpolated error range to zero before this index.", type = int)
    
    parser.add_argument("--gt_path",       default = "", required = True, help = "Input gt data directory", type = str)
    parser.add_argument("--gt_name",       default = "", required = True, help = "Input gt data file names", type = str)
    parser.add_argument("--gt_label",      default = "", required = True, help = "GT plotting label", type = str)
    parser.add_argument("--json_name",     default = "precomputed.json",  help = "Name of the JSON file for caching", type = str)
    parser.add_argument("--subf_pattern",  default = [], nargs = "*", help = "Pattern of the sub folders", type = str)
    parser.add_argument("--subf_npattern", default = [], nargs = "*", help = "Pattern to ignore of the sub folders", type = str)
    parser.add_argument("--exts",          default = [], required = True, nargs = "*", help = "Extensions for different folders", type = str)
    parser.add_argument("--colors",        default = COLORS, nargs = "*", help = "Color to plot the curves", type = str)

    parser.add_argument("--xlabel",       default = "", help = "Plot x label", type = str)
    parser.add_argument("--ylabel",       default = "", help = "Plot y label", type = str)
    parser.add_argument("--start_time",   default = "", help = "Frame starting time", type = float)
    parser.add_argument("--plot_margin",  default = [], nargs = "*", help = "Margin of the plot", type = float)

    parser.add_argument("--figure_size",   default = [8, 4], nargs = "*", help = "Size of the figure", type = float)
    parser.add_argument("-l", "--load",    default = False, action = "store_true", help = "Whether to load cached from json")
    parser.add_argument("--no_legend",     default = False, action = "store_true", help = "Whether to disable plot legend")
    opts = parser.parse_args()

    crop_info = None
    if opts.crop_rx > 0 and opts.crop_ry > 0:
        crop_info = (opts.crop_x, opts.crop_y, opts.crop_rx, opts.crop_ry)
        CONSOLE.log("Cropping:", crop_info)
    
    json_path = os.path.join(opts.input_dir, opts.json_name)
    if not opts.load:
        gt_trans = read_transient(opts.gt_path, opts.gt_name, opts.num_transient, crop_info, extension = opts.ext, identity = False)
        if opts.qnt > 0.1:
            quantile = np.quantile(gt_trans, opts.qnt)
            CONSOLE.log(f"{opts.qnt:.3f} quantile: {quantile:.5f}")
            gt_trans /= quantile
        num_transient, _, _, _ = gt_trans.shape
        gt_curve = gt_trans.mean(axis = (-1, -2, -3))
        
        json_cache_dict = {
            "version": VERSION,
            "crop": crop_info,
            "qnt" : opts.qnt,
            opts.gt_label: {
                "data": gt_curve.tolist(),
                "path":  opts.gt_path,
                "name":  opts.gt_name,
                "ext":   opts.ext
            },
            "cached": []
        }
    else:
        with open(json_path, 'r', encoding = 'utf-8') as file:
            json_cache_dict = json.load(file)
            CONSOLE.log(f"Load from json file at path '{json_path}'")
        gt_curve = np.float32(json_cache_dict['Ground Truth']['data'])
        num_transient = gt_curve.shape[0]

    actual_time = num_transient - 1
    
    plt.figure(figsize = opts.figure_size)
    sns.set(style="whitegrid")
    
    font_path = '../utils/font/biolinum.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    # Set the loaded font as the global font
    plt.rcParams['font.family'] = prop.get_name()
    plt.rc('font', size=14)       
    plt.rc('axes', labelsize=15)  
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15) 
    plt.rc('legend', fontsize=13) 
    plt.rc('figure', titlesize=12)
    plt.tight_layout()
    plt.subplots_adjust(*opts.plot_margin)     # bt 0.13 for single line ticks, 0.17 for double line ticks
    xs = np.linspace(0, actual_time, num_transient)
    
    if not opts.load:
        for i, (folder, name, legend, ext) in enumerate(zip(opts.folders, opts.input_names, opts.legends, opts.exts)):
            input_path = os.path.join(opts.input_dir, folder)

            # first, list all the sub-folders that contains HDR image file
            all_sub_folders = os.listdir(input_path)
            folders, non_prefix_folders = enumerate_filtered_folders(all_sub_folders, opts.subf_pattern, opts.subf_npattern, path_prefix = input_path)

            all_results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers = MAX_WORKERS) as executor:
                futures = [executor.submit(process_frame, folders, name, ext, i, crop_info) for i in range(num_transient)]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                    if num_transient and len(all_results) >= num_transient:
                        break
            CONSOLE.log(f"Folder '{folder}' data loading completed.")
            all_results = natsort.natsorted(all_results, key = lambda x: x[0])

            # This can be used to draw fluctuation plot
            frame_vals = np.stack([arr for _, arr in all_results]) 

            json_cache_dict["cached"].append({
                "label": legend,
                "data": frame_vals.tolist(),
                "path": input_path,
                "name": name,
                "ext":  ext,
            })

            mean_val = frame_vals.mean()
            frame_vals *= gt_curve.mean() / mean_val
            plot_smooth_error_curve(xs, frame_vals.mean(axis = -1), frame_vals, legend, opts.colors[i + 1], 
                                    fine_num = opts.interp_num, nullify_point = opts.nullify_point)
    else:
        for i, cached in enumerate(json_cache_dict["cached"]):
            frame_vals = np.float32(cached["data"])
            mean_val = frame_vals.mean()
            frame_vals *= gt_curve.mean() / mean_val
            plot_smooth_error_curve(xs, frame_vals.mean(axis = -1), frame_vals, cached["label"], opts.colors[i + 1], 
                                    fine_num = opts.interp_num, nullify_point = opts.nullify_point, alpha = 0.4)
    
    if not opts.load:
        with open(json_path, 'w', encoding = 'utf-8') as file:
            json.dump(json_cache_dict, file, indent = 4)
            CONSOLE.log(f"Cached to json file at path '{json_path}'")
            
    plt.scatter(xs, gt_curve, s = 5, c = opts.colors[0])
    plt.plot(xs, gt_curve, label = f'{opts.gt_label}', c = opts.colors[0])
    
    if not opts.no_legend:
        plt.legend()
        
    plt.xlim((0, actual_time))
    plt.xlabel(opts.xlabel)
    plt.ylabel(opts.ylabel)

    tick_pos, labels = double_row_formatter(num_transient, opts.start_time, opts.time_length, opts.sol)
    plt.xticks(tick_pos, labels = labels)
    plt.grid(axis = 'both', visible = True)
    plt.savefig('curve_output.png', dpi = 400)
