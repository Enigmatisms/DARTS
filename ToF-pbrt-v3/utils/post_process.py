""" Post processing for images
- Merge several exr images
- median filtering
"""

import os
import re
import sys
import tqdm
import imageio
sys.path.append("..")
import cv2 as cv
import numpy as np
import taichi as ti
import configargparse

from taichi.math import vec3
from natsort import natsorted
from utils.shot_noise_remove import filtering
from utils.pfm_reading import read_pfm, write_pfm
from temporal.transient_read import get_transient_exr
from multiprocessing import Pool
from functools import partial
from rich.console import Console
CONSOLE = Console(width = 128)

EXR_PATTERN = re.compile(r'^\w+_\d+\.exr$')
PFM_PATTERN = re.compile(r'^\w+_\d+\.pfm$')

def get_viz_options():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",       is_config_file = True, help='Config file path')
    parser.add_argument("-f", "--input_folder", default = "", help = "Folder of input files. When present, we might not need to fill in input_files.", type = str)
    parser.add_argument("-i", "--input_files",  default = [], help = "Input file paths (for multiple comparison)", nargs='*', type = str)
    parser.add_argument("-w", "--weights",      default = [], help = "Input file paths (for multiple comparison)", nargs='*', type = float)
    parser.add_argument("-q", "--quantile",     default = 0., help = "Normalize the output picture with its <x> quantile value (for display)", type = float)
    parser.add_argument("-th", "--threshold",   default = 0.2, help = "Outlier threshold", type = float)
    parser.add_argument("-op", "--out_path",    default = "", help = "Output file path", type = str)
    parser.add_argument("-on", "--out_name",    default = "", help = "Output file name", type = str)
    parser.add_argument("-m",  "--mode",        required = True, choices = ["gated", "transient"], help = "Processing mode.", type = str)
    parser.add_argument("--flip",               action = 'store_true', help = "Flipping PFM file")
    parser.add_argument("--prefilter",          action = 'store_true', help = "Prefiltering images before merging")
    parser.add_argument("--allow_npy",          action = 'store_true', help = "Whether to add npy file to input")
    parser.add_argument("--save_hdr",           action = 'store_true', help = "Whether to save HDR file after quantiling normalization (for pfm only)")
    parser.add_argument("--extra_output",       default = -1, help = "Except from merge all the images, we can merge fewer and output", type = int)    
    # consider the problem of blowing up the memory
    parser.add_argument("--pattern",   default = [], nargs="*", type = str, help = "Pattern that should be found in 'folder' names, the folders with all patterns can be used")
    parser.add_argument("--npattern",  default = [], nargs="*", type = str, help = "Pattern that can't be found in 'folder' names, the folders with all patterns can't be used")
    return parser.parse_args()

@ti.kernel
def filtering(src: ti.template(), dst: ti.template(), threshold: float):
    # This filtering should be improved.
    for i, j in dst:
        center_pix = src[i + 1, j + 1]
        valid = False
        pix_val_sum = vec3([0, 0, 0])
        for k_x in range(3):
            for k_y in range(3):
                if k_x == 1 and k_y == 1: continue
                pix = src[i + k_x, j + k_y]
                norm = (pix - center_pix).norm()
                if norm < threshold:    # if the current pixel has similiar adjacent pixel: meaning that the pixel is not shot noise
                    valid = True
                    break
                pix_val_sum += pix
        if valid:
            dst[i, j] = center_pix
        else:
            dst[i, j] = pix_val_sum / 8.
            
def enumerate_filtered_folders(all_folders: list, patterns: list, npattern: list, path_prefix = None, sort = False):
    folders = []
    non_prefix_folders = []
    for folder in all_folders:
        actual_path = folder if path_prefix is None else os.path.join(path_prefix, folder) 
        if os.path.isfile(actual_path): continue
        if patterns:
            continue_flag = False
            for pattern in patterns:        # check for all patterns
                if pattern not in folder: 
                    continue_flag = True
                    break
            if continue_flag: continue
        if npattern:
            continue_flag = False
            for pattern in npattern:        # check for all negative patterns
                if pattern in folder: 
                    continue_flag = True
                    break
            if continue_flag: continue
        if path_prefix is not None:
            folders.append(actual_path)
            non_prefix_folders.append(folder)
        else:
            folders.append(folder)
    if sort:
        # Note that, for floating digits (less than 10), sorted is better (natsorted will lead to strange result)
        return sorted(folders), sorted(non_prefix_folders)
    return folders, non_prefix_folders

def enumerate_filtered_files(folder_path, cat_self = False, sorted = False):
    all_files = os.listdir(folder_path)
    results = []
    for file in all_files:
        if not EXR_PATTERN.match(file) and not PFM_PATTERN.match(file): 
            continue            # only xxxx_{some digit}.exr will be saved
        if cat_self:
            results.append(os.path.join(folder_path, file))
        else:
            results.append(file)
    if sorted:
        return natsorted(results)
    return results

def half_resize(img):
    return cv.resize(img, (img.shape[1] >> 1, img.shape[0] >> 1))
            
def merge_transients(opts):
    """ Merge the whole transient profile (several folders) 
        - Get all folder and all possible files (names)
    """
    if not opts.input_folder:
        raise ValueError("Input folder not specified.")
    out_folder_name = opts.out_name
    if not out_folder_name:
        CONSOLE.log(f"[yellow]Warning[/yellow]: Output folder name is not set. Default 'temporary' will be used.")
        out_folder_name = 'temporary'
    all_folders = os.listdir(opts.input_folder)
    folders, _ = enumerate_filtered_folders(all_folders, opts.pattern, opts.npattern)
    if not folders: 
        CONSOLE.log("[yellow]Warning[/yellow]: No valid folder found. Exiting...")
        exit(0)
    # enumerate all files (and possibly avoid steady state exr file)
    item_set = set()
    item_set_built = False
    exr_pattern = re.compile(r'^\w+_\d+\.exr$')
    pfm_pattern = re.compile(r'^\w+_\d+\.pfm$')
    for folder in folders:
        path = os.path.join(opts.input_folder, folder)
        all_files = os.listdir(path)
        for file in all_files:
            if not exr_pattern.match(file) and not pfm_pattern.match(file): continue            # only xxxx_{some digit}.exr will be saved
            if item_set_built and file not in item_set:
                CONSOLE.log(f"[yellow]Warning[/yellow]: file '{file}' in folder '{folder}' is not seen before.")
            else:
                item_set.add(file)
        if item_set_built == False:
            item_set_built = True
    if not item_set: 
        CONSOLE.log("[yellow]Warning[/yellow]: No valid exr file found. Exiting...")
        exit(0)
    files = natsorted(list(item_set))
    # now we have files and folders
    frames = []
    h, w = 0, 0
    CONSOLE.log(f"Merging and denoising {len(files)} transient frames.")
    
    field_init = False
    img_field = None
    out_field = None
    if opts.threshold > 1e-5:
        ti.init(arch = ti.cpu)
    for hdr_file in tqdm.tqdm(files):
        file_buffer = []
        for folder in folders:
            hdr_path = os.path.join(opts.input_folder, folder, hdr_file)
            image = get_transient_exr(hdr_path, True) if hdr_file.endswith("exr") else read_pfm(hdr_path, opts.flip).astype(np.float32)
            # if 'right' in folder:
            #     image *= 2
            if opts.threshold > 1e-5:
                if not field_init:
                    field_init = True
                    h, w, _ = image.shape
                    img_field = ti.Vector.field(3, float, (h + 2, w + 2))
                    out_field = ti.Vector.field(3, float, (h, w))
                pad_img = np.pad(image, ((1, 1), (1, 1), (0, 0)))
                img_field.from_numpy(pad_img)
                filtering(img_field, out_field, opts.threshold)
                image = out_field.to_numpy()
            file_buffer.append(image)
        averaged = np.stack(file_buffer, axis = 0).mean(axis = 0)
        frames.append(averaged)
        if h == 0:
            h, w, _ = averaged.shape
    num_frames = len(frames)
    if opts.threshold > 1e-5:
        CONSOLE.log("Post-filtering transients...")
        for i in tqdm.tqdm(range(num_frames)):
            pad_img = np.pad(frames[i], ((1, 1), (1, 1), (0, 0)))
            img_field.from_numpy(pad_img)
            filtering(img_field, out_field, opts.threshold)
            frames[i] = out_field.to_numpy()
    
    out_file_folder = opts.out_path
    if out_file_folder == "":
        out_file_folder = opts.input_folder
    
    out_folder = os.path.join(out_file_folder, out_folder_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    CONSOLE.log(f"Exporting merged transients...")
    if opts.quantile > 0.1:
        qnt = np.quantile(frames, opts.quantile)
        CONSOLE.log(f"Transient quantile: {qnt:.4f}")
        for i in tqdm.tqdm(range(num_frames)):
            exr_name = files[i]
            frame    = frames[i]
            no_ext_name = exr_name[:-4]
            out_file_name = os.path.join(out_folder, f"{no_ext_name}.png")
            imageio.imwrite(out_file_name, ((frame / qnt).clip(0, 1) * 255).astype(np.uint8))
            np.save(os.path.join(out_folder, f"{no_ext_name}.npy"), frame)
    CONSOLE.log(f"Transient is merged. Output directory: {out_folder}")
    
def proc_patch(img, threshold = 0.001, mask_low = False):
    qnt_high = np.quantile(img, 1. - threshold)
    mask = (img <= qnt_high)
    if mask_low:
        qnt_low = np.quantile(img, threshold)
        mask   &= img >= qnt_low
    return img[mask], mask

def load_image(zipped, resize, flip):       # a closure
    name, weight = zipped
    if name.endswith("exr"):
        image = get_transient_exr(name, True)
    elif name.endswith("pfm"):
        image = read_pfm(name, flip).astype(np.float32)
    else:
        image = np.load(name)
    if resize: image = half_resize(image)
    return image * weight

def load_images_parallel(images_files, weights, resize, flip, num_processes = 8):
    args_list = list(zip(images_files, weights))
    with Pool(num_processes) as pool:
        func = partial(load_image, resize = resize, flip = flip)
        images = pool.map(func, args_list)
    return images

def image_batch_process(images_files, weights, threshold, verbose = False, resize = False, flip = False, prefilter = False):
    """ Return list of images and (possibly) filtered image 
        threshold: threshold for median filter noise detection
        resize: for large scale comparison, resize will keep the result and also save memory 
        flip: PFM images might need to be horizontally flippped
    """
    images = load_images_parallel(images_files, weights, resize, flip)
    if prefilter and threshold > 1e-5:
        ti.init(arch = ti.cpu)
        h, w, _ = images[0].shape
        img_field = ti.Vector.field(3, float, (h + 2, w + 2))
        out_field = ti.Vector.field(3, float, (h, w))
        for i in range(len(images)):
            pad_img = np.pad(images[i], ((1, 1), (1, 1), (0, 0)))
            img_field.from_numpy(pad_img)
            filtering(img_field, out_field, threshold)
            images[i] = out_field.to_numpy()
    image = np.stack(images, axis = 0).sum(axis = 0)
    if prefilter:
        return images, image
    if threshold > 1e-5:
        ti.init(arch = ti.cpu)
        h, w, _ = image.shape
        img_field = ti.Vector.field(3, float, (h + 2, w + 2))
        out_field = ti.Vector.field(3, float, (h, w))
        pad_img = np.pad(image, ((1, 1), (1, 1), (0, 0)))
        img_field.from_numpy(pad_img)
        filtering(img_field, out_field, threshold)
        image = out_field.to_numpy()
        if verbose:
            CONSOLE.log("Median filtered for the result.")
    return images, image
            
def merging_exr(opts):
    if opts.input_folder:
        all_files = os.listdir(opts.input_folder)
        files = []
        for file in all_files:
            npy_can_input = opts.allow_npy and file.endswith('.npy')
            if not file.endswith(".exr") and not file.endswith(".pfm") and not npy_can_input: continue
            files.append(os.path.join(opts.input_folder, file))
    else:
        files = opts.input_files
    num_files = len(files)
    if num_files == 0:
        raise ValueError("No valid files found. Please check input_folder or input_files")
    
    weights = opts.weights if opts.weights else [1 / num_files for _ in range(num_files)]
    images, image = image_batch_process(files, weights, opts.threshold, flip = opts.flip, prefilter = opts.prefilter)
    
    out_file_folder = opts.out_path
    if out_file_folder == "":
        if opts.input_folder:
            out_file_folder = opts.input_folder
    if out_file_folder == "":
        CONSOLE.log("[yellow]Warning[/yellow]: Output directory not specified. Exiting...")
        exit(0)
    if not os.path.exists(out_file_folder):
        os.makedirs(out_file_folder)
    out_path = os.path.join(out_file_folder, f"{opts.out_name}.png")
    qnt = 0.0
    hdr_path = os.path.join(out_file_folder, "extra_hdr")
    if opts.save_hdr and not os.path.exists(hdr_path):
        os.makedirs(hdr_path)
    if opts.quantile > 0.1:
        qnt = np.quantile(image, opts.quantile)
        imageio.imwrite(out_path, ((image / qnt).clip(0, 1) * 255).astype(np.uint8))
        if opts.save_hdr:
            out_pfm = os.path.join(hdr_path, f"{opts.out_name}.pfm")
            write_pfm(out_pfm, image / qnt)
            CONSOLE.log(f"HDR (pfm format) is stored: {hdr_path}")
    np.save(os.path.join(out_file_folder, f"{opts.out_name}.npy"), image)
    CONSOLE.log(f"EXR is merged. Output directory: {out_file_folder}")
    if opts.extra_output >= 0:
        extra_num = min(2, opts.extra_output)
        for i in range(extra_num + 1):
            num_img   = 2 ** i
            extra_img = np.stack(images[:num_img], axis = 0).mean(axis = 0)
            qnt = np.quantile(extra_img, opts.quantile)
            out_path = os.path.join(out_file_folder, f"{opts.out_name}-merge-{num_img}.png")
            imageio.imwrite(out_path, ((extra_img / qnt).clip(0, 1) * 255).astype(np.uint8))
            if opts.save_hdr:
                out_pfm = os.path.join(hdr_path, f"{opts.out_name}-merge-{num_img}.pfm")
                write_pfm(out_pfm, extra_img / qnt)
        CONSOLE.log(f"{extra_num + 1} images are outputed.")
            

if __name__ == "__main__":
    opts = get_viz_options()
    if opts.mode == 'gated':
        merging_exr(opts)
    else:
        merge_transients(opts)
        