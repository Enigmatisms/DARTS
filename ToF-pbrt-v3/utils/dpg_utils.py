""" Dearpygui utilities (from my AnalyticalGuiding repo)
"""
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Iterable

SLIDER_WIDTH = 120

def value_sync(set_tag: str):
    """ Call back function for synchronizing input/slider """
    def value_sync_inner(_sender, app_data, _user_data):
        dpg.set_value(set_tag, app_data)
    return value_sync_inner

def callback_wrapper(fixed_callback, callbacks = None):
    if callbacks is None:
        return fixed_callback
    if isinstance(callbacks, Iterable):
        def callback(_sender, app_data, _user_data):
            fixed_callback(_sender, app_data, _user_data)
            for func in callbacks:
                func()
    else:
        def callback(_sender, app_data, _user_data):
            fixed_callback(_sender, app_data, _user_data)
            callbacks()
    return callback

def create_slider(
    label: str, tag: str, min_v: float, max_v: float, 
    default: float, in_type: str = "float", other_callback = None
):
    """ Create horizontally grouped (and synced) input box and slider """
    slider_func = dpg.add_slider_float if in_type == "float" else dpg.add_slider_int
    input_func  = dpg.add_input_float if in_type == "float" else dpg.add_input_int
    input_callback = callback_wrapper(value_sync(tag), other_callback)
    slider_callback = callback_wrapper(value_sync(f"{tag}_input"), other_callback)
    with dpg.group(horizontal = True):
        input_func(tag = f"{tag}_input", default_value = default, 
                                width = 110, callback = input_callback)
        slider_func(label = label, tag = tag, min_value = min_v,
                    max_value = max_v, default_value = default, width = SLIDER_WIDTH, callback = slider_callback)

def value_updator(config_dict: dict, skip_params: set):
    """ Get values from sliders """
    for attr in config_dict.keys():
        if attr in skip_params: continue
        config_dict[attr] = dpg.get_value(attr)
    return config_dict

def change_color_theme(sender, user_data):
    if user_data or dpg.get_value('dark_theme'):
        dpg.bind_theme("default")
    else:
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # dpg.show_style_editor()
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (197, 197, 197, 216), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 131, 220, 216), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (230, 230, 230), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (254, 254, 254, 70), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (247, 247, 247, 128), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, (160, 160, 160), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (0, 88, 149, 176), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (4, 4, 4), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (130, 183, 238, 200), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (10, 10, 10), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (170, 208, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (0, 129, 218), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255), category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvTheme)
        dpg.bind_theme(global_theme)

def pad_rgba(img: np.ndarray):
    """ Convert RGB images to RGBA images """
    alpha = np.ones((*img.shape[:-1], 1), dtype = img.dtype)
    img = np.concatenate((img, alpha), axis = -1).transpose((1, 0, 2))
    return np.ascontiguousarray(img)
