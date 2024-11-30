""" Calculate variance of the rendered results
    @date: 2023-5-18
"""

import os
import sys
sys.path.append("..")
import json
import natsort
import numpy as np
import matplotlib.pyplot as plt
import configargparse
import cv2 as cv

from pathlib import Path
from exr_read import read_exr
from typing import Tuple, List
from utils.pfm_reading import read_pfm
from utils.post_process import proc_patch

from rich.console import Console
CONSOLE = Console(width = 128)

names = ("UDPT", "PP", "PB", "DARTS", "PP-DARTS", "more-1", "more-2", "more-3", "more-4")
SCALING = True

def get_options(delayed_parse = False):
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("--gt",                   default = None, help = "Ground truth file path", type = str)
    parser.add_argument("-j", "--json_input",     required = True, help = "Json input file path (ROI selection)", type = str)
    parser.add_argument("-f", "--flip",           default = False, action = 'store_true', help = "Flipping pfm image")
    parser.add_argument("-is", "--input_folders", default = None, help = "Input file paths (for multiple comparison)", nargs='*', type = str)
    parser.add_argument("-ip", "--input_prefix",  default = None, help = "Input file paths prefix", type = str)
    parser.add_argument("-id", "--idx",           default = -1, help = "Scale amplitude to the level of <id> folders", type = int)
    parser.add_argument("--gt_multiplier",        default = 16, help = "GT rendering time multiplier", type = int)
    parser.add_argument("-q", "--quantile",       default = 0., help = "Normalize the output picture with its <x> quantile value", type = float)
    parser.add_argument("--copy_print",           default = False, action = 'store_true', help = "Print format better suitable for markdown table")
    parser.add_argument("--print_within",         default = False, action = 'store_true', help = "Print within-image variance")
    parser.add_argument("--rescale_mean",         default = False, action = 'store_true', help = "Rescale image groups to the same mean value")
    parser.add_argument("--outlier_reject",       default = False, action = 'store_true', help = "Use quantiled outlier rejection")
    parser.add_argument("--exr_rescale",          default = False, action = 'store_true', help = "If rescale_mean is True, does the program rescale exr format? False by default.")
    parser.add_argument("--no_json",              default = False, action = 'store_true', help = "If rescale_mean is True, does the program rescale exr format? False by default.")
    parser.add_argument("--save_mse_json",        default = False, action = 'store_true', help = "If True, MSE-time will be saved.")
    parser.add_argument("--output_gt_info",       default = False, action = 'store_true', help = "If True, GT rendering time will be outputed")
    parser.add_argument("--mode",                 default = 'mse', choices = ['mse', 'var'], help = "Comparison choices")
    parser.add_argument("-o", "--json_output",    default = "", help = "Json output file path (Variance record)", type = str)
    if delayed_parse:
        return parser
    return parser.parse_args()

def get_selection(path: str, no_json = False):
    dict_vals = {}
    result = []
    if path and not no_json:
        with open(path, 'r', encoding = 'utf-8') as file:
            dict_vals = json.load(file)
        rects = dict_vals["rects"]
        for rect in rects:
            result.append((tuple(rect["p1"]), tuple(rect["p2"])))
    if dict_vals.get("whole_window", False) or no_json:
        result.append(-1)
    return result

def outlier_rejection_mean(patch: np.ndarray, iter = 2, quantile = 0.98):
    """ Reject firefly induced mean shifting """
    mean = patch.mean()
    for _ in range(iter):
        diff = patch - mean
        diff_norm = np.linalg.norm(diff, axis = -1)
        qnt = np.quantile(diff_norm, quantile)
        mask = diff_norm <= qnt
        mean = patch[mask, :].mean()
    return mean

def scale(img, factor = 1):
    if not SCALING: return img
    new_size = (img.shape[1] >> factor, img.shape[0] >> factor)
    return cv.resize(img, new_size, interpolation = cv.INTER_LINEAR)

def variance_analysis(
    input_list: List[np.ndarray], gt: np.ndarray, 
    roi: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
    copy_print = False, print_within = False, index = -1, time = -1, mode = 'mse'
):
    all_mse = []
    result_mse = []
    if copy_print:
        if index >= len(names):
            name = 'Dummy'
        else:
            name = names[index] if index >= 0 else "\t"
        print(f"|{name}|", end = '')
    for i, ((sx, sy), (ex, ey)) in enumerate(roi):
        gt_patch = scale(gt[sy:ey, sx:ex, :].copy())
        mses = []
        for img in input_list:
            patch = scale(img[sy:ey, sx:ex, :].copy())
            patch, mask = proc_patch(patch)
            mse = (patch - gt_patch[mask]) ** 2            # relative MSE
            mses.append(mse.mean())
        final_mse = np.mean(mses)
        if copy_print:
            print(f"{final_mse:.10f}", end = " |")
        else:
            print(f"For ROI {i + 1}, variance = {final_mse:.6f}")
            if print_within:
                print("Within image variance: ", end = '')
                for mse in mses:
                    print(f"{mse:.5f}", end = ', ')
                print("")
        all_mse.append(mses)
        result_mse.append(final_mse)
    if copy_print:
        if time >= 0:
            print(f"{time:.4f}|\t|")
        else:
            print("\t|\t|")
    # if index == 0:
    #     for mses in all_mse:
    #         indices = np.argsort(mses)
    #         print(indices, np.float32(mses)[indices])
    return all_mse, result_mse

def get_input_images(input_path:str, quantile: float = 0, flip_pfm = False) -> List[np.ndarray]:
    all_names = os.listdir(input_path)

    image_names = []
    extensions = ['exr', 'pfm', 'npy']
    image_format = 'none'
    for ext in extensions:
        image_names  = list(filter(lambda name: name.endswith(f".{ext}"), all_names))
        if len(image_names): 
            image_format = ext
            break
    if len(image_names) == 0:
        raise ValueError(f"Folder '{input_path}' does not contain any valid hdr file.")
    image_names = natsort.natsorted(image_names)
    images = []
    for name in image_names:
        path = os.path.join(input_path, name)
        if image_format == 'exr':
            image = read_exr(path, quantile).astype(float)
        else:
            if image_format == 'pfm':
                image = read_pfm(path, flip_pfm)
            else:
                image = np.load(path)
                if flip_pfm:
                    image = np.flip(image, axis = 1)
            if quantile > 0.1:
                image /= np.quantile(image, quantile)
        images.append(image)
    return images, image_format

def get_avg_time(path: str):
    input_path = os.path.join(path, "time.log")
    if os.path.exists(input_path):
        with open(input_path, 'r', encoding = 'utf-8') as file:
            lines = file.readlines()
            sum_time = 0
            sum_cnt  = 0
            for line in lines:
                if not line: continue
                try:
                    value = float(line[:-1])
                except ValueError:
                    pass
                else:
                    sum_time += value
                    sum_cnt += 1
            if sum_cnt == 0: return -1.
            time = sum_time / sum_cnt
            CONSOLE.log(f"Avg time for this scene: {time:.5f}")
            return time
    else:
        CONSOLE.log(f"Time information not exist in '{input_path}'")
    return -1

def single_calculation():
    """ Calculate variance analysis result just for one folder """
    parser = get_options(True)
    parser.add_argument("-i", "--input_folder",   required = True, help = "Input file path", type = str)
    opts = parser.parse_args()
    roi_selection = get_selection(opts.json_input, opts.no_json)
    gt_image      = read_exr(opts.gt, opts.quantile)
    var_images, _ = get_input_images(opts.input_folder) 
    if gt_image is None:
        gt_image = np.stack(var_images, axis = 0).mean(axis = 0)
    else:
        gt_image = gt_image.astype(float)
    qnt = np.quantile(gt_image, opts.quantile)
    var_images = [img / qnt for img in var_images]
    time          = get_avg_time(opts.input_folder)
    all_mse, result_mse = variance_analysis(var_images, gt_image, roi_selection, opts.copy_print, opts.print_within)
    if opts.json_output:
        out_path = os.path.join(opts.input_folder, opts.json_output)
        with open(out_path, 'w', encoding = 'utf-8') as file:
            json_file = {"name": Path(opts.input_folder).stem, "time": time, "all_mse": all_mse, "mse": result_mse}
            json.dump(json_file, file, indent = 4)
            
def comparison():
    """ Compare the output results of multiple folders
    """
    opts = get_options()
    roi_selection = get_selection(opts.json_input, opts.no_json)
    if opts.gt:
        extra_gt = read_exr(opts.gt, opts.quantile).astype(float)
        CONSOLE.log(f"Using ground truth image loaded from '{opts.gt}'")
    else:
        CONSOLE.log(f"No ground truth image specified")
        extra_gt = None
    image_group = []
    gt_images   = []
    image_fmts  = []
    means       = []
    input_folders = []
    for folder in opts.input_folders:
        if opts.input_prefix is not None:
            input_folders.append(os.path.join(opts.input_prefix, folder))
        else:
            input_folders.append(folder)
    num_folders = len(input_folders)
    if num_folders < 2:
        raise ValueError(f"Input folders has length {num_folders}, which is less than 2")
    times = []
    for name in input_folders:
        parts = name.split("/")
        CONSOLE.log(f"Candidate {parts[-1] if len(parts[-1]) else parts[-2]} is being loaded.")
        var_images, ifmt = get_input_images(name, 0, opts.flip) 
        if ifmt == 'pfm' and opts.rescale_mean == False:
            CONSOLE.log(f"Folder '{name}' contains 'pfm' HDR file, `rescale_mean` is set to be {True}")
            setattr(opts, "rescale_mean", True)
        image_fmts.append(ifmt)
        gt_image         = np.stack(var_images, axis = 0).mean(axis = 0)
        # this quantiling is going to cause some troubles
        if opts.mode == 'var':
            qnt = np.quantile(gt_image, opts.quantile)
            var_images = [img / qnt for img in var_images]
        mean_val = outlier_rejection_mean(gt_image) if opts.outlier_reject else gt_image.mean()
        means.append(mean_val)
        image_group.append(var_images)
        gt_images.append(gt_image)
        time = get_avg_time(name)
        times.append(time)
    if roi_selection[-1] == -1:
        CONSOLE.log("Whole window added.")
        h, w, _ = gt_images[0].shape
        roi_selection[-1] = ((0, 0), (w, h))
        
    CONSOLE.log("Mean values:", means)
    gt_mean = means[opts.idx] if extra_gt is None else extra_gt.mean()
    if opts.rescale_mean:        # only old variance comparison mode will rescale mean value
        if opts.idx == -1:
            raise ValueError("For mode 'var', mean-rescaling index can not be -1. Set `idx` in the config.")
        for i in range(num_folders):
            if image_fmts[i] == 'exr' and not opts.exr_rescale: continue
            ratio = gt_mean / means[i]      # rescaling ratio
            means[i] *= ratio
            gt_image[i] *= ratio
            num_images = len(image_group[i])
            for j in range(num_images):
                image_group[i][j] *= ratio
        CONSOLE.log("Mean values after scaling:", means)
    use_rule = len(roi_selection) > 1 and not opts.copy_print
    result_mses = []
    for i, (gt, var_images) in enumerate(zip(gt_images, image_group)):
        if use_rule:
            print(f"=================== Config {i + 1} ===================")
        # in order to use the mean value as gt for out method, we use opts.idx to specify
        if extra_gt is not None:
            gt2use = extra_gt
        else:
            gt2use = gt if (opts.idx == -1) or (opts.mode == 'var') else gt_images[opts.idx]
        _, result_mse = variance_analysis(var_images, gt2use, roi_selection, opts.copy_print, opts.print_within, i, time = times[i], mode = opts.mode)
        result_mses.append(result_mse)
    mses = np.float32(result_mses)
    
    diff_ratio = mses.min(axis = 0) / mses.max(axis = 0)
    # diff_ratio = mses[1] / mses[0]
    print("|best ratio|", end = '')
    for x in diff_ratio:
        print(f"{x:.6f}", end = ' |')
    print("\t|")
    
    if opts.save_mse_json:
        mses = np.concatenate([mses, np.float32(times).reshape(-1, 1)], axis = -1)
        if opts.output_gt_info:
            gt_info = np.full((1, mses.shape[-1]), -1, dtype = mses.dtype)
            gt_info[0, -1] = mses[opts.idx, -1] * opts.gt_multiplier
            mses = np.concatenate([mses, gt_info], axis = 0)
        
        mse_output_dir = os.path.join(Path(opts.json_input).parent, "mse-time.json")
        with open(mse_output_dir, 'w', encoding = 'utf-8') as file:
            json.dump(mses.tolist(), file, indent = 4)
            CONSOLE.log(f"MSE-time cache file stored to: {mse_output_dir}")


if __name__ == "__main__":
    comparison()
