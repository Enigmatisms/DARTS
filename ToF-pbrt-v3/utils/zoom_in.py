""" Magnified image comparison utility function
    @date 2023.9.25
    actually, the reason I am not using a class (but this ugly global dict) is that
    this code is adopted from the previous code of mine and I am too lazy to improve the structure
"""

import os
import sys
import cv2 as cv
sys.path.append("..")
import json

import tqdm
import shutil
import numpy as np
import configargparse
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from pathlib import Path
from pfm_reading import read_pfm
from rich.console import Console
from temporal.transient_read import get_transient_exr
from dpg_utils import create_slider, value_updator, change_color_theme

OUTPUT_EXT = 'png'
BORDER_SZ  = -8

bright_theme  = True
mouse_enabled = False 

images = []
image_names = []
rects  = []
colors = [(53, 162, 159), (255, 207, 150), (79, 112, 156), (33, 53, 85)]             # preset colors
rect_tag_set = []

CONSOLE = Console(width = 128)

"""
    All these things should be done tonight
    The thing you should carefully consider for maximum eifficiency:
    - How you are going to load the image(s)? How to store and organize the selected area?
        - [ ] we store the selection area that can be reused for multiple different images in one json file
        - [ ] and we add the image name (and folders) in the jsno file for the post process script to read
            - [ ] stop using folder + image name, use a complete name for input
            - [ ] for each image, we should generate a folder containing the cropped area and the original image containing the frames
            - [ ] There will be a top-level folder containing the output json (for regeneration)
    - How to output the selected images:
        - [ ] different selection frame has different color. For different source images: store them in different folders
        - [ ] image frame can be applied before hand (or can be entirely cancelled)
"""

def get_image(names, flip_pfm = False):
    global images
    file_type = None
    record_h, record_w = 0, 0
    for file_name in names:
        file_type = "image"
        if file_name[-3:] not in {'jpg', 'png'}:
            file_type = file_name[-3:]
        if file_type is None or not os.path.exists(file_name):
            raise RuntimeError(f"Valid file (.png / .jpg / .exr / .npy / .pfm) not found: '{file_name}'")
        elif file_type == "image":
            image = plt.imread(file_name)            
        elif file_type == "npy":
            image = np.load(file_name)
        elif file_type == "exr":
            image = get_transient_exr(file_name, True)
        elif file_type == "pfm":
            image = read_pfm(file_name, flip_pfm).astype(np.float32)
        else:
            raise RuntimeError(f"Invalid file format (not among .png / .jpg / .exr / .npy / .pfm): '{file_name}'")
        local_h, local_w, _ = image.shape
        if record_h == 0:
            record_h, record_w = local_h, local_w
        else:
            if record_h != local_h or record_w != local_w:
                raise RuntimeError(f"Images have different sizes: recorded: {(record_w, record_h)}. Current: {(local_w, local_h)}.")
        images.append(image)
        image_names.append(file_name)
    configs['width'] = record_w
    configs['height'] = record_h
    num_images = len(images)
    CONSOLE.log(f"{num_images} image{'' if num_images == 1 else 's'} loaded.")

def get_options():
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("--iname",         required = False, nargs='+', type = str)
    parser.add_argument("--output_folder", required = False, type = str)
    parser.add_argument("--input_json",    default = None, type = str)
    parser.add_argument("--oname", "-no",  default = "output.json", type = str)
    parser.add_argument("--qnt",           default = 0.99, type = float)
    parser.add_argument("--flip_pfm",      default = False, action = 'store_true', help = 'Whether to horizontally flip the PFM format file.')
    return parser.parse_args()
        
def valid_checking(config):
    return config["sw"] > 0 and config["sh"] > 0

def mouse_move_callback():
    global configs
    if valid_checking(configs):         # habe manually specified selection area
        pos_x, pos_y = dpg.get_mouse_pos()
        half_sw = configs["sw"] / 2
        half_sh = configs["sh"] / 2
        pos_x = np.clip(pos_x, half_sw, configs["width"] - half_sw)
        pos_y = np.clip(pos_y, half_sh, configs["height"] - half_sh)
        pmin = np.int32([pos_x - half_sw, pos_y - half_sh]) + configs["border_sz"]
        pmax = np.int32([pos_x + half_sw, pos_y + half_sh]) + configs["border_sz"]
    else:
        if configs["rect_p1"] is None:
            return
        pos_x, pos_y = dpg.get_mouse_pos()
        configs["rect_p2"] = np.int32([pos_x, pos_y])
        pmin = (np.minimum(configs["rect_p1"], configs["rect_p2"]) + configs["border_sz"]).tolist()
        pmax = (np.maximum(configs["rect_p1"], configs["rect_p2"]) + configs["border_sz"]).tolist()
    dpg.configure_item("selection", pmin = pmin, pmax = pmax, show = True)
        
def mouse_pressed_callback(sender, app_data):
    global configs
    if app_data == 0:
        if not valid_checking(configs) and mouse_enabled:
            pos_x, pos_y = dpg.get_mouse_pos()
            configs["rect_p1"] = np.int32([pos_x, pos_y])
            
def mouse_release_callback(sender, app_data):
    """ Set direction here w.r.t the left focus
        We will record a direction and plot it on the screen
    """
    global configs
    manual_select = valid_checking(configs)
    
    # if we single press the lbutton without moving the mouse, we don't want the rectangle to be recorded
    if not manual_select and configs["rect_p2"] is None: 
        configs["rect_p1"] = None
        return
    if app_data == 0:       # 1 might be right button, I need left button
        if manual_select:         # manually specified selection area
            if mouse_enabled == False: return
            pos_x, pos_y = dpg.get_mouse_pos()
            half_sw = configs["sw"] / 2
            half_sh = configs["sh"] / 2
            pos_x = np.clip(pos_x, half_sw, configs["width"] - half_sw)
            pos_y = np.clip(pos_y, half_sh, configs["height"] - half_sh)
            configs["rect_p1"] = np.int32([pos_x - half_sw, pos_y - half_sh])
            configs["rect_p2"] = np.int32([pos_x + half_sw, pos_y + half_sh])
        create_new_rect()
        dpg.configure_item("selection", show = False)
    else:
        if configs["rect_p1"] is not None:
            configs["rect_p1"] = None
            dpg.configure_item("selection", show = False)
    
def updator_callback():
    global configs
    sw = dpg.get_value("sw")
    sh = dpg.get_value("sh")
    val = min(sh, sw)
    dpg.set_value("sw", val)
    dpg.set_value("sh", val)
    
def clear_setting_callback():
    global configs
    configs["sw"] = 0
    configs["sh"] = 0
    dpg.set_value("sw", 0)
    dpg.set_value("sh", 0)

def info_callback():
    global configs
    image_id = configs['image_id']
    print(f"images[{image_id}].name = {image_names['image_id']}")

def focus_callback_constructor(set_value):
    def focus_callback(val = set_value):
        global mouse_enabled
        mouse_enabled = val
    return focus_callback

def mouse_enable_callback():
    global mouse_enabled
    mouse_enabled = True
    
def mouse_disable_callback():
    global mouse_enabled, configs
    mouse_enabled = False
    configs["rect_p1"] = None
    dpg.configure_item("selection", show = False)
    
def create_new_rect():
    global configs, rect_tag_set
    num_rects = len(rect_tag_set)
    color = colors[num_rects % 4]
    tag = f"rect_{num_rects:02d}"
    rect_tag_set.append(tag)
    pmin = np.minimum(configs["rect_p1"], configs["rect_p2"])
    pmax = np.maximum(configs["rect_p1"], configs["rect_p2"])
    rects.append((pmin, pmax))
    pmin = (pmin + configs['border_sz']).tolist()
    pmax = (pmax + configs['border_sz']).tolist()
    configs["rect_p1"] = None
    configs["rect_p2"] = None
    if dpg.does_item_exist(tag):
        dpg.configure_item(item = tag, pmin = pmin, pmax = pmax, show = True, color = color)
    else:
        dpg.draw_rectangle(tag = tag, pmin = pmin, pmax = pmax, show = True, color = color, parent = 'display', thickness = 2)
    
def get_folder_name(string: str):
    parts = string.split("/")
    if len(parts) > 3:
        parts = parts[-3:]
    for keyword in ("exr", "outputs", "output"):
        try:
            parts.remove(keyword)
        except ValueError:
            pass
    if len(parts) > 3:
        CONSOLE.log("[yellow] Warning: The filename still has more than 3 items: ", parts)
    return f"{parts[-2]}-{Path(parts[-1]).stem}"

def make_color_padding(image: np.ndarray, color: tuple, size = 2):
    h, w, _ = image.shape
    framed_image = np.tile((np.float32(color) / 255)[None, None, ...], (h + size * 2, w + size * 2, 1))
    framed_image[size:-size, size:-size, :] = image
    return framed_image
    
def export_image_json():
    """ exporting cropped images and json file """
    out_folder = configs["out_dir"]
    json_name  = os.path.join(out_folder, f"{configs['out_name']}.json")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        if configs['del_dir']:
            shutil.rmtree(out_folder)
            os.makedirs(out_folder)
    # exporting cropped image patches and the image painted with selection frame
    for image_origin, image_name in zip(images, image_names):
        image = image_origin.copy()
        if configs['quantile'] > 0.1:
            image /= np.quantile(image, configs['quantile'])
        child_folder = get_folder_name(image_name)
        folder_name = os.path.join(out_folder, child_folder)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # output all the cropped file
        image_output = (image.copy()[..., ::-1].clip(0, 1) * 255.).astype(np.uint8)          # account for the channel reverse for stupid opencv
        for i, rect in enumerate(rects):
            min_x, min_y = rect[0]
            max_x, max_y = rect[1]
            image_cropped = image[min_y:max_y, min_x:max_x, :].clip(0, 1)
            out_image_name = os.path.join(folder_name, f"patch_{i + 1:2d}.{OUTPUT_EXT}")
            framed_patch = make_color_padding(image_cropped, colors[i])
            plt.imsave(out_image_name, framed_patch.clip(0, 1))
            image_output = cv.rectangle(image_output, (min_x, min_y), (max_x, max_y), colors[i][::-1], 3)
        image_w_frame = os.path.join(folder_name, f"selection_frames.{OUTPUT_EXT}")
        cv.imwrite(image_w_frame, image_output)
    # exporting json file
    
    with open(json_name, 'w', encoding = 'utf-8') as file:
        json_file = {"rects": [], "image_names": [], "quantile": configs["quantile"], "whole_window": True}
        for i, (p1, p2) in enumerate(rects):
            min_pt = np.minimum(p1, p2)
            max_pt = np.maximum(p1, p2)
            json_file["rects"].append({"p1": min_pt.tolist(), "p2": max_pt.tolist(), "id": i + 1})
        for image_name in image_names:
            json_file["image_names"].append(image_name)
        json.dump(json_file, file, indent = 4)
    CONSOLE.log(f":star2: Images & json file are exported. Output folder: '{out_folder}'")
    
        
def pop_rect():
    global rect_tag_set
    if len(rects) > 0:
        label = rect_tag_set.pop()
        dpg.configure_item(label, show = False)
        rects.pop()
        
def key_callback(sender, app_data):
    """ Keyboard responses to be used """
    global configs
    if app_data == 69:      # E (export)
        CONSOLE.log("Json file and images are exported.")
        export_image_json()    
    elif app_data == 80:    # P (pop)
        pop_rect()    
    elif app_data == 72:    # H help
        CONSOLE.rule(title = 'Help on Keyboard Settings')
        CONSOLE.log("Press 'E' to export image and selection json")
        CONSOLE.log("Press 'P' to pop a selection rect")
        CONSOLE.log("Press [SPACE] to switch between images")
        CONSOLE.log("Press [ESC] to quit without saving")
    elif app_data == 32:    # switch background images
        configs['image_id'] = (configs['image_id'] + 1) % len(images)
        dpg.set_value("image_id", configs['image_id'])
    elif app_data == 256:   # ESC
        if configs["rect_p1"] is not None:
            configs["rect_p1"] = None
            dpg.configure_item("selection", show = False)
        else:
            dpg.stop_dearpygui()

if __name__ == "__main__":
    CONSOLE.log("Selection zoom-in and exporting utilities. Press 'H' to get help on keyboard settings.")
    args = get_options()
    qnt = args.qnt
    names = args.iname
    if args.input_json is not None:
        CONSOLE.log(f":earth_asia: Loading saved json file from '{args.input_json}'")
        with open(args.input_json, 'r') as json_file:
            json_dict = json.load(json_file)
        names = json_dict["image_names"]
        qnt   = json_dict["quantile"]
        
    configs = {
        "width"       :0,
        "height"      :0,
        "quantile"    :args.qnt,
        "rect_p1"     :None,
        "rect_p2"     :None,
        "sw"          :100,
        "sh"          :100,
        "border_sz"   :BORDER_SZ,
        "image_id"    :0,
        "out_dir"     :args.output_folder,
        "out_name"    :args.oname,
        "del_dir"     :False
    }
    get_image(names, args.flip_pfm)
    skip_params = {"width", "height", "rect_p1", "rect_p2", "border_sz", "out_dir", "out_name"}

    dpg.create_context()
    change_color_theme(None, not bright_theme)

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width = configs['width'], height = configs['height'], 
                            default_value = images[0], format = dpg.mvFormat_Float_rgb, tag = "image")

    with dpg.window(label="Image Selection", tag = "display", no_bring_to_front_on_focus = True):
        dpg.add_image("image")
        dpg.draw_rectangle(pmin = (0, 0), pmax = (0, 0), tag = "selection", color = (0, 200, 0), show = False)
        
    if args.input_json is not None:                         # load saved json file
        for i, rect in enumerate(json_dict["rects"]):
            tag = f"rect_{i:02d}"
            rect_tag_set.append(tag)
            rects.append((rect["p1"], rect["p2"]))
            pmin = (np.int32(rect["p1"]) + BORDER_SZ).tolist()
            pmax = (np.int32(rect["p2"]) + BORDER_SZ).tolist()
            dpg.draw_rectangle(tag = tag, pmin = pmin, pmax = pmax, show = True, color = colors[i % 4], parent = 'display', thickness = 2)
            
    with dpg.window(label="Control panel", tag = "control"):
        create_slider("quantile", "quantile", 0.0, 1.0, 0.99)
        with dpg.group(horizontal = True):
            dpg.add_input_int(label = "select w", tag = "sw", width = 100, default_value = configs['sw'])
            dpg.add_input_int(label = "select h", tag = "sh", width = 100, default_value = configs['sh'])
            
        with dpg.group(horizontal = True):
            dpg.add_input_int(label = "Image ID", tag = "image_id", width = 100, min_value = 0, max_value = len(images) - 1, min_clamped = True, max_clamped = True)
            dpg.add_button(label = "Show Info", tag = "show_info", width = 100, callback = info_callback)
            
        with dpg.group(horizontal = True):
            dpg.add_button(label = 'Save Json', tag = 'json_save', width = 100, callback = export_image_json)
            dpg.add_button(label = 'Clear Manual', tag = 'clear', width = 100, callback = clear_setting_callback)
            dpg.add_checkbox(label = 'Dark theme', tag = 'dark_theme', 
                             default_value = False, callback = change_color_theme)
    
    with dpg.item_handler_registry(tag="handler1") as handler:
        dpg.add_item_focus_handler(callback=mouse_enable_callback)
    with dpg.item_handler_registry(tag="handler2") as handler:
        dpg.add_item_focus_handler(callback=mouse_disable_callback)
        
    dpg.bind_item_handler_registry("display", "handler1")
    dpg.bind_item_handler_registry("control", "handler2")

    with dpg.handler_registry():
        dpg.add_key_release_handler(callback=key_callback)
        dpg.add_mouse_move_handler(callback=mouse_move_callback)
        dpg.add_mouse_release_handler(callback=mouse_release_callback)
        dpg.add_mouse_click_handler(callback=mouse_pressed_callback)
    dpg.create_viewport(title='Image Selection', width = configs["width"] + 25, height = configs["height"] + 25)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        configs = value_updator(configs, skip_params)

        raw_data = images[configs['image_id']].copy()
        if configs["quantile"] > 0.1:
            qnt = np.quantile(raw_data, configs["quantile"])
            raw_data /= np.quantile(raw_data, configs["quantile"])
        dpg.configure_item("image", default_value = raw_data)

    dpg.destroy_context()
