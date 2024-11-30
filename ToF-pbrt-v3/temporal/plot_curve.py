import os
import tqdm
import shutil
import imageio
import numpy as np
import configargparse
import matplotlib.pyplot as plt
from transient_read import read_transient
from shot_noise_removal import shot_peaks_detection, local_median_filter

def get_options():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("-nt", "--num_transient", required= True,           type = int, help = "Number of transient frames.")
    parser.add_argument("-tid", "--time_id",       required= True,          type = int, choices=[0, 1], help = "Time log id")
    parser.add_argument("-did", "--display_id",    required= True,          type = int, nargs="+", help = "Id for curves to be displayed")
    parser.add_argument("-f", "--filter",          default = False,         action='store_true', help = "Curves to be displayed.")
    parser.add_argument("-qnt", "--quantile",      default = 0,             type = float, help = "Quantiling number")
    parser.add_argument("--folder",    default = None,          type = str, help = "The folder to be listed, all the folders in this folder will be enumerated for zoom_in.py")
    parser.add_argument("--pattern",   default = [], nargs="+", type = str, help = "Pattern that should be found in 'folder' names, the folders with all patterns can be used")
    parser.add_argument("--npattern",  default = [], nargs="+", type = str, help = "Pattern that can't be found in 'folder' names, the folders with all patterns can't be used")
    parser.add_argument("--file_name", default = "result",      type = str, help = "The name of the file for each folder. A common name is assumed for all folders")
    parser.add_argument("--crop_sx",   default = 0, help = "Cropping parameter: start x", type = int)
    parser.add_argument("--crop_sy",   default = 0, help = "Cropping parameter: start y", type = int)
    parser.add_argument("--crop_ex",   default = 0, help = "Cropping parameter: end x", type = int)
    parser.add_argument("--crop_ey",   default = 0, help = "Cropping parameter: end y", type = int)
    
    parser.add_argument("--output",    default = "",      type = str, help = "Output file folder")
    return parser.parse_args()

if __name__ == "__main__":
    opts = get_options()
    crop_info = None
    if opts.crop_ex > 0 and opts.crop_ey > 0:
        crop_info = (opts.crop_sx, opts.crop_sy, opts.crop_ex, opts.crop_ey)
        print("Cropping:", crop_info)
    
    folders = os.listdir(opts.folder)
    all_trans  = []
    curves = []
    for folder in folders:
        if os.path.isfile(folder): continue
        if opts.pattern:
            continue_flag = False
            for pattern in opts.pattern:        # check for all patterns
                if pattern not in folder: 
                    continue_flag = True
                    break
            if continue_flag: continue
            for pattern in opts.npattern:        # check for all patterns
                if not pattern: break
                if pattern in folder: 
                    continue_flag = True
                    break
            if continue_flag: continue
        file_path = os.path.join(opts.folder, folder)
        trans = read_transient(file_path, opts.file_name, opts.num_transient, crop_info, identity = False)
        curve = trans.mean(axis = tuple(i for i in range(1, trans.ndim)))
        curves.append(curve)
        all_trans.append(trans)
    for i in range(1, len(all_trans)):
        all_trans[0] += all_trans[i]
    gt_images = all_trans[0] / len(all_trans)
        
    all_curves = np.stack(curves, axis = 0)
    if opts.quantile > 0.1:
        all_curves /= np.quantile(all_curves, opts.quantile)
        gt_images  = ((gt_images / np.quantile(gt_images, opts.quantile)).clip(0, 1) * 255).astype(np.uint8)
    gt_curve = all_curves.mean(axis = 0)
    
    if opts.output and opts.quantile > 0.1:
        num_images = gt_images.shape[0]
        if not os.path.exists(opts.output):
            os.makedirs(opts.output)
        else:
            shutil.rmtree(opts.output)
            os.makedirs(opts.output)
        for i in tqdm.tqdm(range(num_images)):
            output_name = os.path.join(opts.output, f"{i:04d}.png")
            imageio.imwrite(output_name, gt_images[i, ...])
            
    print(gt_curve.shape, all_curves.shape)
    if opts.filter:
        print("Curve filtering is ON.")
        params = {'prominence': 0.00, 'threshold': 0.08}
        peaks = shot_peaks_detection(gt_curve, params)
        local_median_filter(gt_curve, peaks)
    ts = np.arange(len(gt_curve))
    plt.plot(ts, gt_curve, label = 'ground truth')
    for idx in opts.display_id:
        plt.plot(ts, all_curves[idx], label = f'realization {idx+1}')
    plt.xlabel('time')
    plt.ylabel('temporal response')
    plt.legend()
    plt.grid(axis = 'both')
    plt.show()
    