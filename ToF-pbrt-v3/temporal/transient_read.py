import re
import os
import sys
import taichi as ti
sys.path.append("..")
import imageio
import numpy as np 
import OpenEXR, Imath
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
from temporal.opts import get_tdom_options
from temporal.shot_noise_removal import shot_peaks_detection, local_median_filter
from utils.pfm_reading import read_pfm
from utils.shot_noise_remove import filtering

from rich.console import Console
CONSOLE = Console(width = 128)

colors = ("#DF7857", "#4E6E81", "#F99417")

def get_avg_time(path: str, verbose = False):
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
            if verbose:
                CONSOLE.log(f"Avg time for this scene: {time:.5f}")
            return time
    else:
        if verbose:
            CONSOLE.log(f"Time information not exist in '{input_path}'")
    return -1

def extract_inner_name(string: str):
    all_parts = string.split("/")
    if all_parts[-1]:                       # if path is xxx/yyy/, then the [-1] element is '', which is useless
        return all_parts[-1]
    return all_parts[-2]

def median_filter(x: np.ndarray, radius = 2):
    h, w, layers = x.shape
    for layer in range(layers):
        for ri in range(h):
            for ci in range(w):
                block = x[max(ri - radius, 0):min(ri + radius + 1, h), max(ci - radius, 0):min(ci + radius + 1, w), layer]
                value = x[ri, ci, layer]
                if np.isinf(value) or np.isnan(value):
                    median = np.median(block)
                    if np.isinf(median):
                        CONSOLE.log("Warning: median value is inf:", block.ravel())
                        median = 0.
                    x[ri, ci, layer] = median  # median value
    return x

def get_transient_exr(path: str, numerical_guard = False):
    inputFile = OpenEXR.InputFile(path)
    pixelType = Imath.PixelType(Imath.PixelType.HALF)
    dataWin = inputFile.header()['dataWindow']
    imgSize = (dataWin.max.y - dataWin.min.y + 1, dataWin.max.x - dataWin.min.x + 1)
    tmp = list(inputFile.header()['channels'].keys())
    if(len(tmp) != 3):
        prog = re.compile(r"\d+")
        channels = np.array(np.argsort([int(re.match(prog, x).group(0)) for x in tmp], -1, 'stable'))
        channels[0::3], channels[2::3] = deepcopy(channels[2::3]),deepcopy(channels[0::3])
        tmp = np.array(tmp)
        tmp = tmp[list(channels)]
    else:
        tmp = np.array(tmp)
        tmp[0], tmp[2] = tmp[2], tmp[0]

    transients = inputFile.channels(tmp, pixelType)
    transients = [np.reshape(np.frombuffer(transients[i], dtype=np.float16), imgSize) for i in range(len(transients))]
    transients = np.stack(transients, axis=2)
    h, w, _ = transients.shape
    transients = transients.reshape(h, w, -1, 3)
    if transients.shape[-2] == 1:
        transients = transients.squeeze(axis = -2)
    if numerical_guard:
        invalid_flags = np.isinf(transients) | np.isnan(transients)
        transients[invalid_flags] = 0
    return transients.astype(np.float32)

def read_hdr_file(file_name, extension, flip = False):
    """ Read HDR file with different extensions """
    if extension == 'exr':
        return get_transient_exr(file_name)
    elif extension == 'pfm':
        return read_pfm(file_name, flip)
    else:
        image = np.load(file_name)
        if flip:
            image = np.flip(image, axis = 1)
        return image

def read_transient(
    input_path: str, input_name: str, num_transient: str, 
    crop_info: tuple = None, identity = True, extension = 'exr', flip = False
):

    all_transients = []
    if num_transient == -1:
        if extension == 'auto':
            all_exrs = list(filter(lambda x: x.endswith(".exr"), os.listdir(input_path)))
            all_pfms = list(filter(lambda x: x.endswith(".pfm"), os.listdir(input_path)))
            all_npys = list(filter(lambda x: x.endswith(".npy"), os.listdir(input_path)))
            len_exr, len_pfm, len_npy = len(all_exrs), len(all_pfms), len(all_npys)
            max_len = max([len_exr, len_pfm, len_npy])
            if len_pfm == max_len:
                extension = 'pfm'
                all_names = all_pfms
            elif len_exr == max_len:
                extension = 'exr'
                all_names = all_exrs
            else:
                extension = 'npy'
                all_names = all_npys
        else:
            all_names = list(filter(lambda x: x.endswith(f".{extension}"), os.listdir(input_path)))
        num_transient = len(all_names)
        if identity == False:
            if f'{input_name}.{extension}' in all_names: 
                num_transient -= 1
    for i in tqdm(range(num_transient)):
        file_name = os.path.join(input_path, f"{input_name}_{i:04d}.{extension}")
        if not os.path.exists(file_name):
            file_name = os.path.join(input_path, f"{input_name}_{i}.{extension}")
        if not os.path.exists(file_name) :
            CONSOLE.log(f"Warning: Please check the input file name, multiple misses: '{input_name}'")
            file_name = os.path.join(input_path, f"{input_name}.{extension}")
        all_transients.append(read_hdr_file(file_name, extension, flip))
    all_transients = np.stack(all_transients, axis = 0)
    CONSOLE.log(f"All transient has: {np.isnan(all_transients).sum()} nans and {np.isinf(all_transients).sum()} infs")
    CONSOLE.log(f"Transient stats: {all_transients.max()}, {all_transients.min()}, {all_transients.mean()}")
    num_infs = np.isinf(all_transients).sum() + np.isnan(all_transients).sum()
    CONSOLE.log(f"Invalid ratio: {num_infs} / {all_transients.size} ({num_infs / all_transients.size * 100} %)")
    CONSOLE.log(f"Zero ratio: {(all_transients == 0).sum()} / {all_transients.size} ({(all_transients == 0).sum() / all_transients.size * 100} %)")
    all_transients[np.isinf(all_transients)] = 0
    if crop_info is not None:
        if len(crop_info) > 4:
            sx, sy, ex, ey = crop_info[:-1]
            if ex == 0 or ey == 0:
                return all_transients
        else:
            sx = crop_info[0] - crop_info[2]
            ex = crop_info[0] + crop_info[2] + 1
            sy = crop_info[1] - crop_info[3]
            ey = crop_info[1] + crop_info[3] + 1
        all_transients = all_transients[:, sy:ey, sx:ex]
    return all_transients

def get_processed_curves(opts, transients, num_transient, qnt = 0.0, filtering = False):
    actual_time = (opts.time_length if opts.time_length > 0 else num_transient) / opts.sol
    xs = np.linspace(0, actual_time, num_transient)
    plt.title("PBRT simulation raw data curve")

    if opts.window_mode == 'diag_tri':
        curves /= curves.max(axis = 1, keepdims = True)
    elif opts.window_mode == 'diag_side_mean':
        side_curves = (curves[0] + curves[2]) / 2.
        curves /= side_curves.max()
        curves = np.float32([curves[0], curves[1]])
    else:       # whole
        curves = transients.mean(axis = (1, 2, 3))     # spatial average
    params = {'prominence': 0.00, 'threshold': 0.08}
    if curves.ndim == 1:
        curves = curves[None, :]
    if filtering:
        for i in range(curves.shape[0]):
            peaks = shot_peaks_detection(curves[i], params)
            local_median_filter(curves[i], peaks)
    if qnt > 0.1:
        qnt_value = np.quantile(curves, qnt)
        curves /= qnt_value
        CONSOLE.log(f"Quantile: {qnt_value}")
    else:
        CONSOLE.log("No quantile")
    return xs, curves, actual_time

if __name__ == "__main__":
    opts = get_tdom_options()
    crop_info = None
    if opts.crop_rx > 0 and opts.crop_ry > 0:
        crop_info = (opts.crop_x, opts.crop_y, opts.crop_rx, opts.crop_ry)
        CONSOLE.log("Cropping:", crop_info)
    
    all_trans = read_transient(opts.input_path, opts.input_name, opts.num_transient,
                               identity = False, crop_info = crop_info, extension = opts.ext, flip = opts.flip)

    img_field = None
    out_field = None
    if opts.ratio_threshold > 0:
        ti.init(arch = ti.cpu)
        _, h, w, c = all_trans.shape
        img_field = ti.Vector.field(3, float, (h + 2, w + 2))
        out_field = ti.Vector.field(3, float, (h, w))
    
    num_transient, h, w, _ = all_trans.shape
    
    qnt = np.quantile(all_trans, opts.qnt)
    CONSOLE.log(f"Quantile: {qnt}")
    output_path = os.path.join(opts.input_path, "outputs")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    runtime = get_avg_time(opts.input_path)
    if runtime > 0:
        CONSOLE.log(f"Runing time: {runtime:.5f}")
    if not opts.skip_output:
        for i in tqdm(range(num_transient)):
            output_name = os.path.join(output_path, f"{i:04d}.png")
            image = all_trans[i, ...] / qnt
            if opts.ratio_threshold > 0:
                img_qnt = np.quantile(image, opts.qnt)
                pad_img = np.pad(image, ((1, 1), (1, 1), (0, 0)))
                img_field.from_numpy(pad_img)
                filtering(img_field, out_field, img_qnt * opts.ratio_threshold)
                image = out_field.to_numpy()
            imageio.imwrite(output_name, (image.clip(0, 1) * 255).astype(np.uint8))
    else:
        CONSOLE.log("Skipping outputing images...")

    if opts.skip_analysis:
        CONSOLE.log("Skipping temporal analysis. Exiting...")
        exit(0)
    
    xs, curves, actual_time = get_processed_curves(opts, all_trans, num_transient)
    
    for i in range(curves.shape[0]):
        # plt.scatter(xs[peaks], curves[i, peaks], s = 40, facecolors = 'none', edgecolors = colors[(i + 1) % 3])   # visualizing shot noise peaks
        plt.scatter(xs, curves[i], s = 5, c = colors[i])
        plt.plot(xs, curves[i], label = f'window[{i+1}]', c = colors[i])
        np.save(f"./cached/curve-{i + 1}.npy", curves[i])
    
    plt.legend()
    plt.xlim((0, actual_time))
    plt.xlabel("Time (ns)")
    plt.grid(axis = 'both')
    plt.show()
    