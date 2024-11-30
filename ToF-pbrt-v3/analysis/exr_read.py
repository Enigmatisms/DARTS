""" Read EXR image
    @date: 2023-5-17
"""

import os
import configargparse
import numpy as np 
import OpenEXR, Imath
import matplotlib.pyplot as plt

from rich.console import Console
CONSOLE = Console(width = 128)

__all__ = ["read_exr"]

def get_options():
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Config file path')
    parser.add_argument("-i", "--input"   , required = True, help = "Input file path/name", type = str)
    parser.add_argument("-o", "--output"  , default = "image.png", help = "Output file path/name", type = str)
    parser.add_argument("-q", "--quantile", default = 0., help = "Normalize the output picture with its <x> quantile value", type = float)
    return parser.parse_args()

def read_exr(input_path: str, quantile: float = 0., clamp = False) -> np.ndarray:
    """ Reading from exr image and apply possible rescaling (by quantile)
    """
    if input_path.lower() == "none" or os.path.exists(input_path) == False:
        return None
    if input_path.endswith(".exr") == False:
        if input_path.endswith(".npy"):
            print("read_exr: Input EXR is actually an numpy npy file.")
            image = np.load(input_path)
        else:
            raise ValueError(f"File extension '{input_path[-4:]}' not supported, should be .exr or .npy")
    else:
        inputFile = OpenEXR.InputFile(input_path)
        pixelType = Imath.PixelType(Imath.PixelType.HALF)
        dataWin = inputFile.header()['dataWindow']
        imgSize = (dataWin.max.y - dataWin.min.y + 1, dataWin.max.x - dataWin.min.x + 1)
        tmp = list(inputFile.header()['channels'].keys())
    
        tmp = np.array(tmp)
        tmp[0], tmp[2] = tmp[2], tmp[0]
    
        channels = inputFile.channels(tmp, pixelType)
        images = [np.reshape(np.frombuffer(channels[i], dtype=np.float16), imgSize) for i in range(len(channels))]
        image = np.stack(images, axis=2)
    invalid_flags = np.isinf(image) | np.isnan(image)
    image[invalid_flags] = 0
    if quantile > 0.1:
        qnt = np.quantile(image, quantile)
        image /= qnt
        if clamp:
            image = np.clip(image, 0, 1)
    return image

if __name__ == '__main__':
    opts = get_options()
    image = read_exr(opts.input, opts.quantile)
    plt.imsave(opts.output, image)