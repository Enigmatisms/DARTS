import os
import json
import argparse

mapping = {'p_cnt': 'photon_count', 'vp_cnt': 'volume_photon_count', 'vg_radius': 'volume_gather_radius'}

def get_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",  "--file", required = True, help = "Input json to modify.", type = str)
    parser.add_argument("--tw",          default = None, help = "Time width", type = float)
    parser.add_argument("--at",          default = None, help = "Allowed rendering time (in seconds).", type = float)
    parser.add_argument("--start_time",  default = None, help = "Rendering starting time (in seconds).", type = float)
    parser.add_argument("--sigma_s",     default = None, help = "Scattering coefficient.", type = float)
    parser.add_argument("--p_cnt",       default = None, help = "Photon cnt.", type = int)
    parser.add_argument("--vp_cnt",      default = None, help = "Volume photon cnt.", type = int)
    parser.add_argument("--vg_radius",   default = None, help = "Volume photon gather radius.", type = float)
    parser.add_argument("--elliptical",  default = None, help = "Elliptical sampling",  type = int)
    parser.add_argument("--da_distance", default = None, help = "DA distance sampling", type = int)
    return parser.parse_args()
    
if __name__ == "__main__":
    opts = get_configs()
    with open(opts.file, 'r', encoding = 'utf-8') as file:
        inputs = json.load(file)
        
    tw = float(inputs["integrator"]["transient_time_width"])
    if opts.tw is not None:
        tw = opts.tw
    
    time_center = float(inputs["integrator"]["transient_time_center"])
    if opts.start_time is not None:
        time_center = opts.start_time + tw / 2
        
    if opts.at is not None:
        inputs["renderer"]["timeout"]                 = f"{opts.at / 60}" # seconds to minutes
    
    if inputs["integrator"]["type"] != "path_tracer":
        inputs["integrator"]["photon_count"]          = int(inputs["integrator"]["photon_count"])
        inputs["integrator"]["volume_photon_count"]   = int(inputs["integrator"]["volume_photon_count"])
    inputs["integrator"]["transient_time_center"] = time_center
    inputs["integrator"]["transient_time_width"]  = tw
    inputs["renderer"]["overwrite_output_files"]  = False
    media = inputs["media"]
    if len(media) == 0:
        print("Warning: this scene has no scattering media")
    else:
        if opts.sigma_s is not None:
            media[0]["sigma_s"] = opts.sigma_s
    for key, json_name in mapping.items():
        value = getattr(opts, key)
        if value is None: continue
        inputs["integrator"][json_name] = value
    if opts.elliptical  is not None:
        inputs["integrator"]["enable_elliptical"] = True if opts.elliptical else False
    if opts.da_distance is not None:
        inputs["integrator"]["enable_guiding"]    = True if opts.da_distance else False
    with open(opts.file, 'w', encoding = 'utf-8') as file:
        json.dump(inputs, file, indent = 4)