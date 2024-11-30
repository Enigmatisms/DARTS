import cv2 as cv
import numpy as np
import configargparse
import matplotlib.pyplot as plt

def get_options():
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("--img1",  required = True, type = str, help = "Path to input image 1")
    parser.add_argument("--img2",  required = True, type = str, help = "Path to input image 2")
    parser.add_argument("--ratio",  default = 2,     help = 'Slash ratio', type = float)
    parser.add_argument("--noflip", default = False, action = 'store_true', help = 'Whether to disable horizontally flipping the slash.')

    return parser.parse_args()

def generate_mask(w: int, h: int, ratio: float, flip = False):
    origin_h = int(h / ratio)
    right_half = np.triu(np.ones((origin_h, origin_h), dtype = np.float32))
    if flip:
        left_half = np.flip(right_half, axis = 1)
        right_half = np.ones((origin_h, origin_h), dtype = np.float32) - left_half
    else:
        left_half  = np.ones((origin_h, origin_h), dtype = np.float32) - right_half
    left_half = cv.resize(left_half, (origin_h, h), interpolation = cv.INTER_NEAREST)
    right_half = cv.resize(right_half, (origin_h, h), interpolation = cv.INTER_NEAREST)
    half_w = w >> 1
    start_x = half_w - (origin_h >> 1)
    left_mask  = np.zeros((h, w), dtype = np.float32)
    right_mask = np.zeros((h, w), dtype = np.float32)
    left_mask[..., start_x:start_x+origin_h]  = left_half
    left_mask[..., :start_x] = 1
    right_mask[..., start_x:start_x+origin_h] = right_half
    right_mask[..., start_x+origin_h:] = 1
    return left_mask[..., None], right_mask[..., None]

if __name__ == "__main__":
    opts = get_options()
    img1 = plt.imread(opts.img1)
    img2 = plt.imread(opts.img2)
    print(img1.dtype, img1.max())
    h, w, _ = img1.shape
    
    left_mask, right_mask = generate_mask(w, h, opts.ratio, not opts.noflip)
    
    result = (left_mask * img1 + right_mask * img2).clip(0, 1)
    plt.imsave("./slash_merged.png", result)
