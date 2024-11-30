""" Post processing for images
- Merge several exr images
- median filtering
"""

import os
import re
import sys
import json
import imageio
sys.path.append("..")
import numpy as np
import taichi as ti
import configargparse

from taichi.math import vec3
from utils.shot_noise_remove import filtering
from utils.pfm_reading import read_pfm, get_title_str
from temporal.transient_read import get_transient_exr
from multiprocessing import Pool
from functools import partial
from rich.console import Console
CONSOLE = Console(width = 128)

EXR_PATTERN = re.compile(r'^\w+_\d+\.exr$')
PFM_PATTERN = re.compile(r'^\w+_\d+\.pfm$')

def get_title_str(folder_name):
    mapping = {"pt-0.2-0.02":"Scattering = 0.2 m^-1, gatewidth = 0.0667 ns", 
               "pt-0.2-0.1":"Scattering = 0.2 m^-1, gatewidth = 0.334 ns", 
               "pt-0.4-0.02":"Scattering = 0.4 m^-1, gatewidth = 0.0667 ns", 
               "pt-0.4-0.1":"Scattering = 0.4 m^-1, gatewidth = 0.334 ns"
    }
    return mapping[folder_name]

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
    parser.add_argument("-m",  "--mode",        required = True, choices = ["gated", "transient", "export"], help = "Processing mode.", type = str)
    parser.add_argument("--flip",               action = 'store_true', help = "Flipping PFM file")
    parser.add_argument("--prefilter",          action = 'store_true', help = "Prefiltering images before merging")
    parser.add_argument("--allow_npy",          action = 'store_true', help = "Whether to add npy file to input")
    parser.add_argument("--save_hdr",           action = 'store_true', help = "Whether to save HDR file after quantiling normalization (for pfm only)")
    parser.add_argument("--extra_output",       default = -1, help = "Except from merge all the images, we can merge fewer and output", type = int)    
    # consider the problem of blowing up the memory
    parser.add_argument("--thresholds",default = [], nargs="*", type = float, help = "Rendering thresholds")
    parser.add_argument("--labels",    default = [], nargs="*", type = str, help = "User specified labels for the sub-folders")
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
            
def get_image_to_display(opts):
    if not opts.input_folder:
        raise ValueError("Input folder not specified.")
    
    folders = opts.input_files
    if not folders: 
        CONSOLE.log("[yellow]Warning[/yellow]: No valid folder found. Exiting...")
        exit(0)
    exr_pattern = re.compile(r'^\w+_\d+\.exr$')
    pfm_pattern = re.compile(r'^\w+_\d+\.pfm$')
    all_sub_folders = set()
    def is_folder(name):
        path = os.path.join(opts.input_folder, folder, name)
        return os.path.isdir(path)
    for folder in folders:
        path = os.path.join(opts.input_folder, folder)
        all_sub_folders = list(filter(is_folder, os.listdir(path)))
    
    field_init = False
    img_field = None
    out_field = None
    if opts.threshold > 1e-5:
        ti.init(arch = ti.cpu)

    all_sub_folders.sort()              # normally it works
    export_file = {"imageBoxes": []}
    for i, sub_folder in enumerate(all_sub_folders):
        subfolder_title = opts.labels[i] if opts.labels else sub_folder
        elements = {"elements": [], "title": f"{opts.out_name} = {subfolder_title}"}
        continue_flag = False
        
        gt_image  = None
        threshold = opts.thresholds[i] if opts.thresholds else opts.threshold
        for folder in folders: 
            folder_path = os.path.join(opts.input_folder, folder, sub_folder)
            if not os.path.isdir(folder_path): 
                continue_flag = True
                break
            
            all_files = os.listdir(folder_path)
            for file in all_files:
                if not exr_pattern.match(file) and not pfm_pattern.match(file): continue            # only xxxx_{some digit}.exr will be saved
                hdr_path = os.path.join(folder_path, file)
                image = load_image((hdr_path, 1), opts.flip)
                if threshold > 1e-5:
                    if not field_init:
                        field_init = True
                        h, w, _ = image.shape
                        img_field = ti.Vector.field(3, float, (h + 2, w + 2))
                        out_field = ti.Vector.field(3, float, (h, w))
                    pad_img = np.pad(image, ((1, 1), (1, 1), (0, 0)))
                    img_field.from_numpy(pad_img)
                    filtering(img_field, out_field, threshold)
                    image = out_field.to_numpy()
                out_folder = os.path.join(opts.out_path, folder, sub_folder)
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                out_file_name = os.path.join(out_folder, f"img.jpg")
                qnt = np.quantile(image, opts.quantile)
                imageio.imsave(out_file_name, ((image / qnt).clip(0, 1) * 255).astype(np.uint8), quality = 96)
                title_str = get_title_str(folder)
                elements["elements"].append({"image": f"./renders/{folder}/{sub_folder}/img.jpg", "title": title_str})
                
                # make ground truth image
                if 'DARTS PT' in title_str:
                    hdr_files = []
                    for hdr_file in all_files:
                        if not exr_pattern.match(hdr_file) and not pfm_pattern.match(hdr_file): continue            # only xxxx_{some digit}.exr will be saved
                        hdr_files.append(os.path.join(folder_path, hdr_file))
                    _, gt_image = image_batch_process(hdr_files, np.ones_like(hdr_files, 
                                dtype = np.float32) / len(hdr_files), threshold, flip = opts.flip, prefilter = opts.prefilter)
                break
        if continue_flag: continue
        CONSOLE.log(f"Sub-folder: '{sub_folder}' finished exporting")
        
        # Exporting GT file
        if gt_image is not None:
            out_folder = os.path.join(opts.out_path, "approx-gt", sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            out_file_name = os.path.join(out_folder, f"img.jpg")
            elements["elements"].append({"image": f"./renders/approx-gt/{sub_folder}/img.jpg", "title": "Approximated GT"})
            qnt = np.quantile(gt_image, opts.quantile)
            imageio.imsave(out_file_name, ((gt_image / qnt).clip(0, 1) * 255).astype(np.uint8), quality = 96)
            
        export_file["imageBoxes"].append(elements)
    json_path = os.path.join(opts.out_path, "../data.js")
    json_data = json.dumps(export_file, indent=4)
    with open(json_path, 'w', encoding = 'utf-8') as file:
        file.write(f'var data = {json_data}')
    
def load_image(zipped, flip):       # a closure
    name, weight = zipped
    if name.endswith("exr"):
        image = get_transient_exr(name, True)
    elif name.endswith("pfm"):
        image = read_pfm(name, flip).astype(np.float32)
    else:
        image = np.load(name)
    return image * weight

def load_images_parallel(images_files, weights, flip, num_processes = 8):
    args_list = list(zip(images_files, weights))
    with Pool(num_processes) as pool:
        func = partial(load_image, flip = flip)
        images = pool.map(func, args_list)
    return images

def image_batch_process(images_files, weights, threshold, verbose = False, flip = False, prefilter = False):
    """ Return list of images and (possibly) filtered image 
        threshold: threshold for median filter noise detection
        resize: for large scale comparison, resize will keep the result and also save memory 
        flip: PFM images might need to be horizontally flippped
    """
    images = load_images_parallel(images_files, weights, flip)
    if prefilter and threshold > 1e-5:
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

if __name__ == "__main__":
    opts = get_viz_options()
    get_image_to_display(opts)
        