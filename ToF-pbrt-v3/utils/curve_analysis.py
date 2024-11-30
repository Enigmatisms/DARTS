""" This script is to evaluate whole images as inputs (different from folder analysis)
    We will input a folder which contains several sub-folders (indicating specific parameter settings)
    and for each folder, we will generate an averaged and denoised GT for all other individual images to calculate MSE
    we will draw a curve according to this parameter change (indicated by folder names) and the variance
"""


import os
import ast
import sys
import json
import imageio
sys.path.append("..")
import numpy as np
import configargparse
import seaborn as sns
import matplotlib.pyplot as plt

from rich.console import Console
from matplotlib import font_manager
from plt_utils import lineplot_with_errbar, try_stack, tolist
from post_process import (
    image_batch_process, enumerate_filtered_folders, 
    half_resize, enumerate_filtered_files, proc_patch
)

CONSOLE = Console(width = 128)
CURRENT_VERSION = '1.2'
WIDE_PLOT = (8, 5)

DRAW_PARAMS = {
    'set_precision', 'plot_title', 'allow_uneven', 
    'extra_gt', 'disable_range', 'extra_labels', 'y_log_scale', 
    'legend_loc', 'wide_plot', 'plot_settings', 'tmfp'
}

def get_viz_options():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",         is_config_file = True, help='Config file path')
    parser.add_argument("-f", "--input_folder",   required = True, help = "Folder of input files. When present, we might not need to fill in input_files.", type = str)
    parser.add_argument("-fs", "--folders",       default = [], nargs="*", type = str, help = "All folder names to be used. input_folder + [folders] + file_name")
    parser.add_argument("-el", "--extra_labels",  default = [], nargs="*", type = str, help = "Extra xtick labels (should match number of folders)")
    parser.add_argument("--plot_settings",        default = [0.11, 0.95, 0.98, 0.17], 
                                                    nargs="*", type = float, help = "Plot settings for the margins.")
    parser.add_argument("-i", "--index_time",     default = 0, help = "To index the time log, use this field.", type = int)
    parser.add_argument("--save_name",            default = "saved.json", help = "Name of the saved json file", type = str)
    
    parser.add_argument("-q", "--quantile",       default = 0., help = "Normalize the output picture with its <x> quantile value (for display)", type = float)
    parser.add_argument("-qn", "--qnt_normalize", default = 0., help = "Normalize the output picture with its <x> quantile value (for display)", type = float)
    parser.add_argument("--tmfp",                 default = 0., help = "TMFP reference value", type = float)
    parser.add_argument("-th", "--threshold",     default = 0.2, help = "Filtering outlier threshold", type = float)
    parser.add_argument("-on", "--out_name",      default = "", help = "Output file name (output .npy file and .png file)", type = str)
    parser.add_argument("-eg", "--extra_gt",      default = "", help = "Extra ground truth file (for convergence analysis, experiment settings are the same)", type = str)
    parser.add_argument("-o", "--output",         action = 'store_true', help = "Whether to output gt npy file and png")
    parser.add_argument("-v", "--violin_plot",    action = 'store_true', help = "Whether to use violin plot as errorbar")
    parser.add_argument("-ndm", "--no_drop_max",  action = 'store_true', help = "(disable) When displaying AMSE distribution, drop the maximum value to keep the figure well-bounded.")
    parser.add_argument("-s", "--save",           action = 'store_true', help = "To boost calculation, we can save the previouly calculated data.")
    parser.add_argument("-l", "--load",           action = 'store_true', help = "Wehther to load pre-computed json file.")
    parser.add_argument("-g",  "--separate_gt",   action = 'store_true', help = "Use the different groups of merged exr image as GT (should usually be False)")
    parser.add_argument("-a",  "--allow_uneven",  action = 'store_true', help = "Allow unevenly distributed xs and try not to use uniform xticks")
    parser.add_argument("-pt", "--plot_title",    action = 'store_true', help = "Whether to plot title")
    parser.add_argument("-dr", "--disable_range", action = 'store_true', help = "Whether to disable min/max range output")
    parser.add_argument("-mm", "--mean_match",    action = 'store_true', help = "Whether to match mean values between images")
    parser.add_argument("--flip",                 action = 'store_true', help = "Flipping the image (PFM image might need this)")
    parser.add_argument("-sp", "--set_precision", action = 'store_true', help = "Set precision (.2f) for xticks")
    parser.add_argument("-yl", "--y_log_scale",   action = 'store_true', help = "Plot log scale y axis")
    parser.add_argument("-w", "--wide_plot",      action = 'store_true', help = "Whether to use wider plots.")
    parser.add_argument("--prefilter",            action = 'store_true', help = "Salient shot noise removal before merging.")
    parser.add_argument("-m",  "--mode",          default = "single", choices = ["multiple", "single"], help = "Processing mode.", type = str)
    parser.add_argument("--legend_loc",           default = "upper left", choices = ["best", "upper left", "upper right", "disable"], help = "Location of the legend.", type = str)
    parser.add_argument("--thresholds",           default = None, type = str, help = "Specifying different filtering threshold for different folders")
    
    # consider the problem of blowing up the memory (probably not goint yo be used)
    parser.add_argument("--pattern",   default = [], nargs="*", type = str, help = "Pattern that should be found in 'folder' names, the folders with all patterns can be used")
    parser.add_argument("--npattern",  default = [], nargs="*", type = str, help = "Pattern that can't be found in 'folder' names, the folders with all patterns can't be used")

    parser.add_argument("--title",  default = "Gatewidth-AMSE curve", help = "Title of the figure.", type = str)
    parser.add_argument("--xlabel", default = "temporal gatewidth", help = "x-label of the figure.", type = str)
    parser.add_argument("--ylabel", default = "AMSE", help = "y-label of the figure.", type = str)
    parser.add_argument("--labels", default = [], nargs="*", help = "Legend labels of the figure.", type = str)
    return parser.parse_args()

def save_to_json(opts, output_folder:str, folder_xs: list, input_data:np.ndarray):
    """ Save configs and precomputed data to json file """
    if not opts.save or opts.load: return
    json_file = {"version": CURRENT_VERSION}
    for key, value in opts._get_kwargs():
        json_file[key] = value
    json_file['folder_xs'] = []
    for x_val in folder_xs:
        if type(x_val) not in {int, float, str}:
            if isinstance(x_val, np.ndarray): x_val = x_val.tolist()
            else: x_val = list(x_val)
        json_file['folder_xs'].append(x_val)
    json_file['input_data'] = tolist(input_data)
    json_path = os.path.join(output_folder, opts.save_name)
    with open(json_path, 'w', encoding = 'utf-8') as file:
        CONSOLE.log(f"Json file outputed to path '{json_path}'.")
        try: 
            json.dump(json_file, file, indent = 4)
        except TypeError:
            CONSOLE.log(f"Json dumpping failed.")
            for key, value in json_file.items():
                print(f"{key} type: {type(value)}")
            exit(1)
        
def load_from_json(opts, load_path: str, skip_params: set):
    """ Load configs and precomputed data from json file """
    with open(load_path, 'r', encoding = 'utf-8') as file:
        json_file = json.load(file)
        version   = json_file.pop('version')
        if version != CURRENT_VERSION:
            raise ValueError(f"Current version '{CURRENT_VERSION}' does not match the version in the saved file '{version}'.")
        folder_xs  = json_file.pop('folder_xs')
        input_data = json_file.pop('input_data')
        try:    # convergence analysis might fail, since input_data is not of the same shape
            input_data = np.float32(input_data)
        except ValueError: pass
        for key, value in json_file.items():
            if key in skip_params: continue
            setattr(opts, key, value)
    return folder_xs, input_data, opts

def npy_loader(path: str, resize = False):
    if not path or not os.path.exists(path) or not path.endswith("npy"):
        CONSOLE.log("No (valid) extra ground truth provided.")
        return None
    CONSOLE.log(f"Extra ground truth loaded from '{path}'")
    gt_img = np.load(path)
    if resize: gt_img = half_resize(gt_img)
    return gt_img

def curve_process(opts, initialize = False, complete_figure = False, draw_now = False):
    input_folders = []
    for folder in opts.folders:
        input_folder = os.path.join(opts.input_folder, folder)
        if not os.path.exists(input_folder):
            CONSOLE.log(f"[yellow]Warning:[/yellow] path '{input_folder}' does not exist.")
        else:
            input_folders.append(input_folder)
    if len(input_folders) == 0:
        if opts.mode == 'single':
            input_folders.append(opts.input_folder)
        else:
            raise ValueError("There is no valid folder in <folders>. Check --folders parameter.")
    CONSOLE.log(f"Input folders: {len(input_folders)} folder(s)")
    nonp_folders = None
    extra_gt     = npy_loader(opts.extra_gt, True)
    gt_images    = []
    input_data   = []
    load = opts.load
    for i, input_folder in enumerate(input_folders):
        if load: 
            load_path = os.path.join(input_folder, opts.save_name)
            if os.path.exists(load_path):
                nonp_folders, all_mses, opts = load_from_json(opts, load_path, DRAW_PARAMS)
                input_data.append(all_mses)
                continue
            CONSOLE.log(f"Valid json file in '{input_folder}' not found. Recomputing...")
        all_folders = os.listdir(input_folder)
        folders, nonp_folders = enumerate_filtered_folders(all_folders, opts.pattern, opts.npattern, path_prefix = input_folder, sort = True)
        if not folders:
            raise ValueError("No valid folders found. Please check input_folder or input_files")
        all_mses = []           # (N_folder, N_images)      -> estimate of the variance (std)
        for j, folder in enumerate(folders):
            all_image_files = enumerate_filtered_files(folder, cat_self = True)
            weights = np.ones_like(all_image_files, dtype = np.float32) / len(all_image_files)
            weight_value = weights[0]
            threshold = opts.thresholds[nonp_folders[j]] if opts.thresholds is not None else opts.threshold
            # only PFM file will be flipped, since pbrt-v3 sets the image orientation
            images, mean_image = image_batch_process(all_image_files, weights, threshold, False, True, opts.flip, opts.prefilter)
            images = np.stack(images, axis = 0) / weight_value          # rescale to the original amplitude
            if opts.qnt_normalize > 0.1:
                qnt_imgs = np.quantile(images, opts.qnt_normalize)
                images     /= qnt_imgs
                mean_image /= qnt_imgs
                CONSOLE.log(f"Image quantiled normalization: {qnt_imgs:.5f}.")
            if opts.separate_gt:            # actually, after taking average, whether to separate or not does not matter this much (difference is small)
                gt_image = mean_image
            else:
                if i > 0: 
                    gt_image = gt_images[j] if extra_gt is None else extra_gt
                else:
                    gt_image = mean_image if extra_gt is None else extra_gt
                    gt_images.append(mean_image)
            if opts.mean_match:             # Variance will induce mean shifts. Photon based methods does not have the same mean value, either
                gt_mean  = gt_image.mean()
                cur_mean = mean_image.mean()
                images *= gt_mean / cur_mean
            # note that mean image serves as ground truth
            mse_pool = []
            for image in images:
                image, mask = proc_patch(image)
                mse_pool.append(((image - gt_image[mask]) ** 2).mean())
            all_mses.append(np.float32(mse_pool))                                            # all_mse is of shape (N --- number of images) 
            if opts.output:
                out_path = os.path.join(folder, f"{opts.out_name}.png")
                if opts.quantile > 0.1:
                    qnt = np.quantile(mean_image, opts.quantile)
                    imageio.imwrite(out_path, ((mean_image / qnt).clip(0, 1) * 255).astype(np.uint8))
                np.save(os.path.join(folder, f"{opts.out_name}.npy"), mean_image)
                CONSOLE.log(f"EXR is merged. Output directory: {folder}")
        # we should be able to specify GT image, since the original GT is not good enough
        input_data.append(try_stack(all_mses))
        save_to_json(opts, input_folder, nonp_folders, input_data[-1])
        CONSOLE.log(f"Input folder '{input_folder}' completed.")
    if initialize:
        if opts.wide_plot: plt.figure(figsize = (8, 5))
        plt.tight_layout()
        l, r, t, b = opts.plot_settings
        plt.subplots_adjust(left = l, right = r, top = t, bottom = b)     # bt 0.13 for single line ticks, 0.17 for double line ticks
        sns.set(style="whitegrid")
        plt.rc('font', size=16)       
        plt.rc('axes', labelsize=17)  
        plt.rc('xtick', labelsize=16) 
        plt.rc('ytick', labelsize=16) 
        plt.rc('legend', fontsize=15) 
        plt.rc('figure', titlesize=14)
        
        font_path = './font/biolinum.ttf'
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
    lineplot_with_errbar(nonp_folders, input_data, title = opts.title, labels = opts.labels,
                         violin_plot = opts.violin_plot, drop_maximum = not opts.no_drop_max, xlabel = opts.xlabel, 
                         make_even = not opts.allow_uneven, ylabel = opts.ylabel, label_as_tick = True, complete_figure = complete_figure, 
                         draw_now = draw_now, save_image = True, tmfp_ref = opts.tmfp, plot_title = opts.plot_title, legend_loc = opts.legend_loc,
                         set_precision = opts.set_precision, disable_min_max = opts.disable_range, extra_label = opts.extra_labels, ylog_scale = opts.y_log_scale)
    
if __name__ == "__main__":
    opts = get_viz_options()
    if opts.load:
        CONSOLE.log(f"Loading from '{opts.save_name}'.\nNote that loading does not output anything.")
    if opts.thresholds is not None:
        opts.thresholds = ast.literal_eval(opts.thresholds)     # get a dict
    curve_process(opts, True, True, True)
