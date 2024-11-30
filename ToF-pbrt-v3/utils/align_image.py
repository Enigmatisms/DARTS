import os
import json
import imageio
import cv2 as cv
import numpy as np
import configargparse
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
from rich.console import Console

CONSOLE             = Console(width = 128)
SERIF_FONT_PATH     = "./font/biolinum.ttf"     
NON_SERIF_FONT_PATH = "./font/msyh.ttc"     
DEFAULT_HEADERS     = ['Original PT (6k SPP)', 'Photon Points (2D)', 'Photon Beams (1D)', 'DARTS PT (Ours) (6k SPP)', 'DARTS Photon Points (Ours)', 'Ground Truth']
TEASER_HEADERS      = ["Photon Points (2D)", "Photon Planes", "Photon Volumes", "DARTS PP (Ours)", "Ground Truth"]

def parse_args():
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config",          is_config_file = True,    help='Config file path')
    parser.add_argument("-f", "--input_folder",    default = [], nargs = '*', help = "Folder of input files. When present", type = str)
    parser.add_argument("--headers",               default = DEFAULT_HEADERS, nargs = '*', help = "Headers for the images.", type = str)
    parser.add_argument("-i", "--input_path",      required = True,          help = "Path to the folders", type = str)
    parser.add_argument("-o", "--output_name",     default = "packed.png",   help = "Name of the output image", type = str)
    parser.add_argument("-m", "--main_image_name", default = "selection_frames.png", 
                                                                             help = "To index the time log, use this field.", type = str)
    parser.add_argument("-p", "--patch_prefix",    default = "patch_ ",       help = "Name of the saved json file", type = str)
    parser.add_argument("--ext",                   default = "png",          help = "Extension of the image file", type = str)
    parser.add_argument("--mse_json",              default = "",             help = "MSE json file input path", type = str)
    parser.add_argument("--side_notes",            default = "",             help = "Figure side notes", type = str)

    parser.add_argument("--mse_index",             default = -1,             help = "Index for MSE list (-1 means no plotting)")
    parser.add_argument("--pad_size",              default = 3,              help = "Padding size for image patches", type = int)
    parser.add_argument("--h_margin",              default = 10,             help = "Horizontal margin for images", type = int)
    parser.add_argument("--p_margin",              default = 8,              help = "Patch margin for image patches", type = int)
    parser.add_argument("--v_margin",              default = 10,             help = "Vertical margin between main images and patches", type = int)
    parser.add_argument("--relative_mse_idx",      default = -1,             help = "Use relative mse", type = int)
    parser.add_argument("--relative_mse_val",      default = 1,              help = "Relative mse value to use", type = float)
    parser.add_argument("--resize_scale",          default = 1.0,            help = "Scaling factor to rescale the image", type = float)
    parser.add_argument("--time_center",           default = 1.0,            help = "Time center of the time-gated images", type = float)
    parser.add_argument("--sol",                   default = 1.0,            help = "Speed of light", type = float)
    parser.add_argument("--cropping_idxs",         default = [], nargs = '*', help = "Cropping", type = StartEnd)
    
    parser.add_argument("--plot_mse",             action = 'store_true',     help = "Whether to plot MSE")
    parser.add_argument("--last_gt",              action = 'store_true',     help = "Whether the last image is GT")
    parser.add_argument("--smart_color",          action = 'store_true',     help = "Whether to plot text with color that's different from the background")
    parser.add_argument("--filter",               action = 'store_true',     help = "Wehther to apply median filter before plotting.")
    parser.add_argument("--add_header",           action = 'store_true',     help = "Add header labels to all the images.")
    parser.add_argument("--outline_text",         action = 'store_true',     help = "Whether to add outline for the text")
    parser.add_argument("--argmin",               action = 'store_true',     help = "Plot the minimum MSE to be red")
    parser.add_argument("--no_patch",             action = 'store_true',     help = "Disable sub-patches")
    return parser.parse_args()

def StartEnd(s):
    try:
        start_i, end_i = map(int, s.split(','))
        return start_i, end_i
    except:
        print(s)
        raise configargparse.ArgumentTypeError("Coordinates must be (start, end)")

def get_original_statistics(example_image, verbose = False):
    H, W, C = example_image.shape
    if verbose:
        dtype   = example_image.dtype 
        max_val = np.max(example_image)
        CONSOLE.log(f"Image is of shape (W = {W}, H = {H}). Channel = {C}. dtype = {dtype}. Max value = {max_val:.3f}")
    return H, W, C, dtype, max_val

def calculate_total_image_size(opts, H: int, W: int, patch_w: int, header_height: int):
    num_images  = len(opts.input_folder)
    num_margins = num_images + 1
    height      = 2 * opts.v_margin + H + patch_w + opts.p_margin + 2 * opts.pad_size + header_height
    width       = W * num_images + num_margins * opts.h_margin
    return height, width

def make_color_padding(image: np.ndarray, color: tuple, size = 2):
    h, w, _ = image.shape
    framed_image = np.tile(color[None, None, ...], (h + size * 2, w + size * 2, 1))
    framed_image[size:-size, size:-size, :] = image
    return framed_image

def cover_images_on_canvas(opts, final_image, images, patch_images, h: int, w: int, pw: int, header_h: int = 0):
    y_start  = opts.v_margin
    py_start = y_start + h + opts.p_margin + header_h
    pw_w_margin = pw + 2 * opts.pad_size
    for i, (image, (patch1, patch2)) in enumerate(zip(images, patch_images)):
        x_start = (i + 1) * opts.h_margin + w * i
        p2_x_start = x_start + pw_w_margin + opts.p_margin
        pad_patch1 = make_color_padding(patch1, color = patch1[0, 0], size = opts.pad_size)
        pad_patch2 = make_color_padding(patch2, color = patch2[0, 0], size = opts.pad_size)

        final_image[y_start:y_start + h + header_h, x_start:x_start + w, :] = image
        final_image[py_start:py_start + pw_w_margin, x_start:x_start + pw_w_margin, :] = pad_patch1
        final_image[py_start:py_start + pw_w_margin, p2_x_start:p2_x_start + pw_w_margin, :] = pad_patch2
    return final_image

def cover_no_patch_on_canvas(opts, final_image, images, h: int, w: int, header_h: int = 0):
    y_start  = opts.v_margin
    for i, image in enumerate(images):
        x_start = (i + 1) * opts.h_margin + w * i
        final_image[y_start:y_start + h + header_h, x_start:x_start + w, :] = image
    return final_image

def read_image(path, rescale = 1.0, patch_size = None, drop_alpha = True):
    image = imageio.imread(path)
    if rescale < 0.99 or rescale > 1.01 or patch_size is not None:
        if patch_size is None:
            new_w, new_h = int(image.shape[1] * rescale), int(image.shape[0] * rescale)
            image = cv.resize(image, (new_w, new_h))
        else:
            image = cv.resize(image, (patch_size, patch_size))
    if drop_alpha: image = image[..., :3]
    return image

def image_put_text(image: np.ndarray, text: str, position, color, font, outline = False):
    original_dtype = image.dtype
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    if outline:
        color_outline = (255, 255, 255) if max(color) == 0 else (0, 0, 0)
        draw.text(position, text, font = font, fill = color, stroke_width = 2, stroke_fill = color_outline)
    else:
        draw.text(position, text, font = font, fill = color)
    return np.array(img_pil, dtype = original_dtype)

def smart_font_color(opts, image: np.ndarray):
    if not opts.smart_color: return (255, 255, 255)
    half_h = image.shape[0] >> 1
    mean_val = image[2:half_h, 2:-2, ...].mean()
    if mean_val > 150: return (0, 0, 0)
    return (255, 255, 255)

def put_text_on_images(opts, images: list, patches: list, mse: np.ndarray, height: int, two_line: int = False):
    if two_line:
        font = ImageFont.truetype(SERIF_FONT_PATH, int(45 * height / 768))
    else:
        font = ImageFont.truetype(SERIF_FONT_PATH, int(50 * height / 768))
    if opts.relative_mse_idx >= 0:
        if mse.shape[1] > 1:
            mse[:, :-1] /= mse[opts.relative_mse_idx, :-1]
        else:
            mse /= mse[opts.relative_mse_idx]
    else:
        if mse.shape[1] > 1:
            mse[:, :-1] /= opts.relative_mse_val
        else:
            mse /= opts.relative_mse_val
    position = (int(12 * height / 768), int(8 * height / 768))

    if opts.argmin and mse.ndim == 2:
        argmin_idx = (mse[:-1, :-1]).argmin(axis = 0)
    else:
        argmin_idx  = None
        opts.argmin = False
    for i, single_mse in enumerate(mse):
        time_val = None
        if len(single_mse) == 1:
            mse_val = single_mse[-1]
        else:
            mse_val = single_mse[-2]
            time_val = single_mse[-1]
        
        if time_val is not None and time_val < 0: 
            time_val = mse[opts.relative_mse_idx, -1]
        
        if time_val is None:
            text = f"MSE: {mse_val:.2f}×" if mse_val > 0 else ""
        else:
            time_str = format_duration(time_val)
            text = f"MSE: {mse_val:.2f}×\n{time_str}" if mse_val > 0 else f"{time_str}"
        if not text: continue
        if two_line:
            text = f"{TEASER_HEADERS[i]}\n{text}"
        font_color = smart_font_color(opts, images[i]) 
        if opts.argmin:
            if argmin_idx[-1] == i: font_color = (255, 0, 0)
        images[i] = image_put_text(images[i], text, position, font_color, font, opts.outline_text)
        if not patches: continue
        for j in range(2):
            mse_val = single_mse[j]
            text = f"MSE: {mse_val:.2f}×" if mse_val > 0 else ""
            if not text: continue
            font_color = smart_font_color(opts, patches[i][j]) 
            if opts.argmin:
                if argmin_idx[j] == i: font_color = (255, 0, 0)
            patches[i][j] = image_put_text(patches[i][j], text, position, font_color, font, opts.outline_text)
    # if two_line:
    #     images[-1] = image_put_text(images[-1], "Ground Truth", position, font_color, font)
    return images, patches

def image_add_header(opts, images: list, H: int, W: int, max_val: float):
    fixed_text_height = 0
    image_pil = Image.fromarray(images[0])
    font_draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(SERIF_FONT_PATH, int(45 * H / 768))
    text_width, fixed_text_height = font_draw.textsize(DEFAULT_HEADERS[0], font)

    for i in range(len(images)):
        image: np.ndarray = images[i]
        new_image = np.full((H + 6 + fixed_text_height, W, 3), 255 if max_val > 1 else 1, dtype = image.dtype)
        new_image[6 + fixed_text_height:, ...] = image
        image_pil = Image.fromarray(new_image)
        draw = ImageDraw.Draw(image_pil)
        text_width, _ = draw.textsize(opts.headers[i], font)
        x = (W - text_width) // 2
        draw.text((x, 2), opts.headers[i], font=font, fill = (0, 0, 0))
        images[i] = np.array(image_pil, dtype = image.dtype)
    CONSOLE.log("Header added.")
    return fixed_text_height + 6

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    components = []
    if hours > 0:
        components.append(f"{int(hours)}h")
    if minutes > 0:
        components.append(f"{int(minutes)}min")
    if seconds > 0 or not components:
        second = int(round(seconds / 10) * 10)
        if second > 59:
            components.pop()
            components.append(f"{int(minutes) + 1}min")
        else:
            components.append(f"{second}s")
    return ''.join(components)

def add_side_notes(opts, H: int, max_val: float, dtype, height: int = 120):
    """ Add side notes (time point and time center) """
    image_pil = Image.fromarray(np.ones((height, H, 3), dtype = dtype))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(SERIF_FONT_PATH, 45)
    
    text_input = f"{opts.side_notes}{opts.time_center/opts.sol:.2f} ns"
    text_width, fixed_text_height = draw.textsize(text_input, font)
    new_image = np.full((6 + fixed_text_height, H, 3), 255 if max_val > 1 else 1, dtype = dtype)
    
    x = (H - text_width) // 2
    image_pil = Image.fromarray(new_image)
    draw = ImageDraw.Draw(image_pil)
    draw.text((x, 3), text_input, font=font, fill = (0, 0, 0))
    
    image_pil = image_pil.rotate(90, expand=True)
    return np.array(image_pil, dtype = dtype)

if __name__ == "__main__":
    opts = parse_args()
    input_paths = [os.path.join(opts.input_path, path) for path in opts.input_folder]

    images = []
    patches = []

    # load all rescaled images
    for i, path in enumerate(input_paths):
        stem_name = Path(opts.main_image_name).stem
        for ext in ('jpeg', 'jpg', 'png'):
            main_image_path = os.path.join(path, f"{stem_name}.{ext}")
            if os.path.exists(main_image_path):
                img = read_image(main_image_path, opts.resize_scale)
                if opts.cropping_idxs:
                    start_i, end_i = opts.cropping_idxs[i]
                    img = img[:, start_i:end_i, :]
                images.append(img)
                break
        else:
            raise ValueError("Supported extensions are: ('jpeg', 'jpg', 'png'), but nothing could be found.")

    H, W, C, dtype, max_val = get_original_statistics(images[0], True)
    patch_w  = (W - opts.p_margin - 4 * opts.pad_size) >> 1      # patch size for each image
    # load all rescaled patches
    if not opts.no_patch:
        for path in input_paths:
            patches.append([
                read_image(os.path.join(path, f"{opts.patch_prefix}1.{opts.ext}"), patch_size = patch_w),
                read_image(os.path.join(path, f"{opts.patch_prefix}2.{opts.ext}"), patch_size = patch_w)
            ])

    if opts.plot_mse and opts.mse_json:
        json_path = os.path.join(opts.input_path, opts.mse_json)
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                loaded = json.load(file)
                json_file = np.float32(loaded)
            put_text_on_images(opts, images, patches, json_file, H, opts.no_patch)

    header_height = 0
    if opts.add_header:
        header_height = image_add_header(opts, images, H, W, max_val)
    if opts.no_patch:
        patch_w = 0
        opts.p_margin = 0

    height, width = calculate_total_image_size(opts, H, W, patch_w, header_height)     # final image resolution
    final_image   = np.ones((height, width, C), dtype = dtype)
    if max_val > 1.0: final_image *= 255

    if opts.no_patch:
        final_image = cover_no_patch_on_canvas(opts, final_image, images, H, W, header_height)
    else:
        final_image = cover_images_on_canvas(opts, final_image, images, patches, H, W, patch_w, header_height)
        
    if opts.side_notes:
        side_note_image = add_side_notes(opts, height, max_val, dtype = final_image.dtype)
        final_image = np.concatenate([side_note_image, final_image], axis = 1)
    CONSOLE.log(f"Output image shape (x, y): ({width}, {height}). Estimated storage: {(width * height * 3) / 1024 ** 2:.2f} MB (for no compression)")
    out_path = os.path.join(opts.input_path, f"{Path(opts.input_path).stem}-{opts.output_name}")
    imageio.imwrite(out_path, final_image)



    


