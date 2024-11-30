import os
import re
import tqdm
import struct
import imageio
import natsort
import numpy as np
import configargparse
import OpenEXR, Imath
import concurrent.futures
from pathlib import Path
from copy import deepcopy
from rich.console import Console

CONSOLE = Console(width = 128)
MAX_WORKERS = 64

def get_options():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",  is_config_file = True, help='Config file path')
    parser.add_argument("-if", "--input_folder", required = True, help = "Input pfm simulation data folder", type = str)
    parser.add_argument("-i",  "--input_name",    required = True, help = "Input pfm simulation data name", type = str)
    parser.add_argument("-o",  "--output_name",   default = "output", help = "Output png data name", type = str)
    parser.add_argument("-q",  "--qnt",           default = 0.99, help = "Normalizing quantile", type = float)
    parser.add_argument("-n",  "--num_transient", default = -1, help = "Number of transients", type = int)
    parser.add_argument("-f",  "--flip",          action = "store_true", help = "Horiziontal flip the image")
    parser.add_argument("-m",  "--merge",         action = "store_true", help = "Merge stored images and output")
    return parser.parse_args()

def get_transient_exr(path: str):
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
    return transients.astype(np.float32)


def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale

def process_image(opts, name):
    if not opts.input_name in name: return None
    if os.path.isdir(name) or not name.endswith('.pfm'): return None
    path = os.path.join(opts.input_folder, name)
    image = read_pfm(path)
    if opts.flip:
        image = np.flip(image, axis=1)
    return (name, image)

def write_image(i, image, output_path, opts):
    image_path = os.path.join(output_path, f"{opts.output_name}_{i:03d}.png")
    imageio.imwrite(image_path, (image.clip(0, 1) * 255).astype(np.uint8))
    return image_path
    
if __name__ == "__main__":
    opts = get_options()
    image_names = os.listdir(opts.input_folder)
    image_names = natsort.natsorted(image_names)

    CONSOLE.log("Loading images...")
    images = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, opts, name) for name in image_names]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                images.append(result)

            if opts.num_transient > 0 and len(images) >= opts.num_transient:
                break
    CONSOLE.log("Images are loaded.")
    images = natsort.natsorted(images, key = lambda x: x[0])
    images = np.stack([packed[1] for packed in images], axis = 0)
    if opts.qnt > 0.1:
        qnt = np.quantile(images, opts.qnt)
        CONSOLE.log(f"Quantile: {qnt}. Shape: {images.shape}")
        images /= qnt
    else:
        CONSOLE.log("Quantiling is not used.")
    output_path = os.path.join(opts.input_folder, "outputs")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    num_images = images.shape[0]
    
    CONSOLE.log("Exporting images.")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(write_image, i, image, output_path, opts) for i, image in enumerate(images)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            
    if opts.merge:
        merged_image = images.mean(axis = 0)
        image_path = os.path.join(output_path, f"merged.png")
        imageio.imwrite(image_path, (merged_image.clip(0, 1) * 255).astype(np.uint8))
    CONSOLE.log(f"{images.shape[0]} images in total are exported.")
    