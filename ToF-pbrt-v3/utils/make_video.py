import os
import cv2
import tqdm
import numpy as np
import configargparse 
from pathlib import Path
from natsort import natsorted
from rich.console import Console
from PIL import ImageFont, ImageDraw, Image
from align_image import image_put_text

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

SERIF_FONT_PATH     = "./font/biolinum.ttf"     
CONSOLE = Console(width = 128)

def reverse(x):
    if x == 'right':
        return 'left'
    return 'right'

def get_tdom_options(delayed_parse = False):
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",  is_config_file = True, help='Config file path')
    parser.add_argument("--input_folder",  required = True, help = "Folder of the inputs", type = str)
    parser.add_argument("--image_1",       required = True, help = "Input image name 1", type = str, nargs = "*")
    parser.add_argument("--image_2",       required = True, help = "Input image name 2", type = str, nargs = "*")
    parser.add_argument("--out_folder",    default = './video_tmp/', help = "Output video folder", type = str)
    parser.add_argument("--out_name",      default = 'video.mp4',    help = "Output video name", type = str)
    parser.add_argument("--bg_color",      default = 'black',    help = "Back ground color of the video", type = str)
    
    parser.add_argument("--info_1",        required = True, help = "Information to be plotted for image 1", type = str, nargs = "*")
    parser.add_argument("--info_2",        required = True, help = "Information to be plotted for image 2", type = str, nargs = "*")
    
    parser.add_argument("--total_time",    default = 8, help = "Information to be plotted for image 2", type = float)
    parser.add_argument("--move_d",        default = 1, help = "Separation bar moving duration", type = float)
    parser.add_argument("--stop_d",        default = 2, help = "Separation bar stopping duration", type = float)
    parser.add_argument("--first_move_t",  default = 2, help = "Time point for the first move", type = float)
    parser.add_argument("--second_move_t", default = 5, help = "Time point for the second move", type = float)
    parser.add_argument("--third_move_t",  default = 5, help = "Time point for the third move", type = float)
    parser.add_argument("--time_scaler",   default = 1, help = "Temporal dilation or compression scaler", type = float)
    parser.add_argument("--acc_ratio",     default = 0.5, help = "Ratio of acceleration", type = float)
    parser.add_argument("--fps",           default = 24,  help = "FPS", type = int)
    parser.add_argument("--text_margin",   default = 10,  help = "Margin between text lines", type = int)
    parser.add_argument("--font_size",     default = 60,  help = "Font size", type = int)
    
    parser.add_argument("--output_w",      default = 1500,  help = "Output width", type = int)
    parser.add_argument("--output_h",      default = 1000,  help = "Output height", type = int)
    parser.add_argument("--resize_w",      default = 720,  help = "Resized width for a single image", type = int)
    parser.add_argument("--resize_h",      default = 720,  help = "Resized height for a single image", type = int)
    parser.add_argument("--image_margin",  default = 720,  help = "Margin between images", type = int)
    
    parser.add_argument("--repeat_times",  default = 1,  help = "Number of times to repeat the video (for transient rendering)", type = int)
    parser.add_argument("--fps_dilate",    default = 2,  help = "Make the same frame be added multiple times", type = int)
    
    parser.add_argument("--skip_output",   default = False, action = "store_true", help = "Whether to skip outputing images")
    parser.add_argument("--output_image",  default = False, action = "store_true", help = "Check the text of the images")
    parser.add_argument("--resize",        default = False, action = "store_true", help = "Whether to resize")
    parser.add_argument("--no_sliding",    default = False, action = "store_true", help = "Whether to disable sliding separation")
    parser.add_argument("--multi_inputs",  default = False, action = "store_true", help = "Whether to enable multiple inputs")
    
    parser.add_argument("--start_dir",     default  = 'left', choices = ['left', 'right'], help = "Initial moving direction of the separate bar", type = str)
    
    if delayed_parse:
        return parser
    return parser.parse_args()

def get_kinetics(opts, distance, direction = 'left'):
    """ Get velocity and acceleration"""
    m_d = opts.move_d
    acc_d = 0.5 * opts.acc_ratio * m_d
    uniform_d = (1 - opts.acc_ratio) * m_d
    coeff = acc_d * acc_d + acc_d * uniform_d 
    acc = distance / coeff
    # vel = acc * acc_d
    num_move_frames = int(opts.move_d * opts.fps)
    num_accl_frames = int(opts.fps * acc_d)
    acc *= (opts.fps * acc_d) / num_accl_frames
    accs = np.zeros(num_move_frames, dtype = np.float32)
    dir_sign = -1 if direction == 'left' else 1
    accs[:num_accl_frames] = acc * dir_sign
    accs[-num_accl_frames:] = - acc * dir_sign
    return accs
    
def get_position(time, acceleration, initial_position, initial_velocity):
    position = initial_position + initial_velocity * time + 0.5 * acceleration * time ** 2
    new_v = initial_velocity + acceleration * time
    return position, new_v

def images_add_info(opts, images_1: np.ndarray, images_2: np.ndarray, W: int, H: int):
    fixed_text_height = 0
    image_pil = Image.fromarray(images_1[0])
    font_draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(SERIF_FONT_PATH, int(opts.font_size * H / 960))
    margin = H / 960 * opts.text_margin
    _, fixed_text_height = font_draw.textsize("abcdefg-ABCDEFG", font)

    for i in range(len(images_1)):
        multi_line1 = opts.info_1[i].split('/')

        line_start = H - len(multi_line1) * (fixed_text_height + margin)
        for j, line in enumerate(multi_line1):
            # left aligned
            images_1[i] = image_put_text(images_1[i], line, (margin, line_start + j * (fixed_text_height + margin)), (255, 255, 255), font, True)

        if i >= len(images_2): continue
        multi_line2 = opts.info_2[i].split('/')
        line_start = H - len(multi_line2) * (fixed_text_height + margin)
        right_align_x = W - margin
        for j, line in enumerate(multi_line2):
            # right aligned
            text_width, _ = font_draw.textsize(line, font)
            images_2[i] = image_put_text(images_2[i], line, (right_align_x - text_width, line_start + j * (fixed_text_height + margin)), (255, 255, 255), font, True)
    return images_1, images_2

def get_resized_image(opts, name, full_path = False):
    if not full_path:
        for extension in ('jpg', 'png'):
            path = os.path.join(opts.input_folder, f"{name}.{extension}")
            if os.path.exists(path):
                break
        else:
            raise ValueError(f"'{name}' does not match any file in '{opts.input_folder}'")
    else:
        path = name
    img = cv2.imread(path)
    if opts.resize:
        img = cv2.resize(img, (opts.resize_w, opts.resize_h))
    return img

def pack_centering(opts, images, width, height, bg_color = 'black'):
    if bg_color == 'black':
        frame = np.zeros((opts.output_h, opts.output_w, 3), dtype = np.uint8)
    else:
        frame = np.full((opts.output_h, opts.output_w, 3), 255, dtype = np.uint8)
    center_x = opts.output_w >> 1
    center_y = opts.output_h >> 1
    
    num_images = len(images)
    whole_size = num_images * width + (num_images - 1) * opts.image_margin
    start_x  = center_x - (whole_size >> 1)
    start_y  = center_y - (height >> 1)
    for image in images:
        frame[start_y:start_y + height, start_x:start_x + width, :] = image
        start_x += width + opts.image_margin
    return frame

if __name__ == "__main__":
    opts = get_tdom_options()
    time_scaling = opts.time_scaler
    if time_scaling != 1:
        for name in ('move_d', 'stop_d', 'first_move_t', 'second_move_t', 'third_move_t', 'total_time'):
            setattr(opts, name, getattr(opts, name) * time_scaling)
    
    CONSOLE.log(f"Video info: duration = {opts.total_time}s, fps: {opts.fps}s. First move tpoint: {opts.first_move_t}s, second move tpoint: {opts.second_move_t}s")
    CONSOLE.log(f"Moving duration: {opts.move_d}s, stopping duration: {opts.stop_d}s. Time scaling: {time_scaling}x")
    
    if opts.out_folder:
        if not os.path.exists(opts.out_folder):
            os.makedirs(opts.out_folder)
    else:
        opts.out_folder = opts.input_folder
    
    if opts.multi_inputs:
        parent_path1 = os.path.join(opts.input_folder, opts.image_1[0])
        parent_path2 = os.path.join(opts.input_folder, opts.image_2[0])
        image_names_1 = os.listdir(parent_path1)
        image_names_2 = os.listdir(parent_path2)
        images_1 = natsorted([os.path.join(parent_path1, img_name) for img_name in image_names_1])
        images_2 = natsorted([os.path.join(parent_path2, img_name) for img_name in image_names_2])
        width, height = opts.resize_w, opts.resize_h
    else:
        images_1 = [get_resized_image(opts, img_name) for img_name in opts.image_1]
        images_2 = [get_resized_image(opts, img_name) for img_name in opts.image_2]

        height, width, _ = images_1[0].shape
        images_1, images_2 = images_add_info(opts, images_1, images_2, width, height)
        
        if opts.output_image:
            CONSOLE.log(f"Images with info are output to '{opts.out_folder}'")
            path1 = os.path.join(opts.out_folder, f"{Path(opts.out_name).stem}-img-1.png")
            path2 = os.path.join(opts.out_folder, f"{Path(opts.out_name).stem}-img-2.png")
            cv2.imwrite(path1, images_1[0])
            cv2.imwrite(path2, images_2[0])

    if opts.skip_output: 
        CONSOLE.log("Output video is disabled, exiting...")
        exit(0)
    output_width  = opts.output_w
    output_height = opts.output_h
    fps = opts.fps
    
    out_video_path = os.path.join(opts.out_folder, opts.out_name) 
    
    output_video   = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))
    CONSOLE.log(f"Writing videos to '{out_video_path}'...")

    moving_frames  = int(opts.move_d * fps)
    first_move_id  = int(opts.first_move_t * fps)
    first_stop_id  = first_move_id + moving_frames
    second_move_id = int(opts.second_move_t * fps)
    second_stop_id = second_move_id + moving_frames
    third_move_id  = int(opts.third_move_t * fps)
    third_stop_id  = third_move_id + moving_frames
    stop_idxs      = {first_stop_id, second_stop_id, third_stop_id}
    total_frames   = int(opts.total_time * fps) 
    if opts.multi_inputs:
        total_frames = opts.repeat_times

    accs    = []
    acc_ptr  = 0
    velocity = 0
    position = width >> 1
    
    for i in tqdm.tqdm(range(total_frames)):
        frames = []
        if opts.multi_inputs:
            RANGE = tqdm.tqdm(range(len(images_1)))
        else:
            RANGE = range(len(images_1))
        for img_id in RANGE:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # ugly, but whatever. This is one-off
            if opts.multi_inputs:
                cur_img1 = get_resized_image(opts, images_1[img_id], True)
                cur_img2 = get_resized_image(opts, images_2[img_id], True)
                (cur_img1,), (cur_img2,) = images_add_info(opts, [cur_img1], [cur_img2], width, height)
            else:
                cur_img1 = images_1[img_id]
                cur_img2 = images_2[img_id]
            if not opts.no_sliding:
                if i == first_move_id:
                    acc_ptr = 0
                    accs = get_kinetics(opts, (width >> 1) - 1, opts.start_dir)
                elif i == second_move_id:
                    acc_ptr = 0
                    accs = get_kinetics(opts, width - 2, reverse(opts.start_dir))
                elif i == third_move_id:
                    acc_ptr = 0
                    accs = get_kinetics(opts, (width >> 1) - 1, opts.start_dir)
                elif i in stop_idxs:
                    velocity = 0
                    acceleration = 0

                if acc_ptr < len(accs):
                    acceleration = accs[acc_ptr]
                    acc_ptr += 1
                else:
                    acceleration = 0

                position, velocity = get_position(1 / fps, acceleration, position, velocity)
            
            pos_idxs = int(np.clip(position, 0, width))
            frame[:, :pos_idxs, :] = cur_img1[:, :pos_idxs, :]
            img_id = np.clip(img_id, 0, len(images_2) - 1)
            frame[:, pos_idxs:, :] = cur_img2[:, pos_idxs:, :]
            frame[:, pos_idxs:pos_idxs + 1, :] = [255, 255, 255]  # 分割线颜色为白色
            if opts.multi_inputs:
                for _ in range(opts.fps_dilate):
                    output_video.write(pack_centering(opts, [frame], width, height, opts.bg_color))
            else:
                frames.append(frame)
        if opts.multi_inputs: continue
        # post processing, like align / padding
        output_video.write(pack_centering(opts, frames, width, height, opts.bg_color))
    output_video.release()