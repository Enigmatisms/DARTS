""" Sample histogram experiments
    @date:   2024-1-1
"""
import os
import matplotlib
import numpy as np
import seaborn as sns
import configargparse
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter
from transient_read import read_transient
from matplotlib import font_manager
from rich.console import Console

CONSOLE = Console(width = 128)

def get_tdom_options(delayed_parse = False):
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",  is_config_file = True, help='Config file path')
    parser.add_argument("--input_path1",   default = "", required = True, help = "Input simulation data folder", type = str)
    parser.add_argument("--input_path2",   default = "", required = True, help = "Input simulation data folder", type = str)
    parser.add_argument("--input_name1",   default = "foam", help = "Input simulation data name", type = str)
    parser.add_argument("--input_name2",   default = "foam", help = "Input simulation data name", type = str)
    parser.add_argument("--cache",         default = None, help = "Whether to store the histogram", type = str)
    parser.add_argument("--ext",           default = "auto", help = "Input file extension (auto-detection by default)", type = str)
    parser.add_argument("--num_transient", default = 500, help = "Number of transient image", type = int)
    parser.add_argument("--crop_start",    default = 60, help = "Transient frame cropping start", type = int)
    parser.add_argument("--crop_end",      default = 40, help = "Transient frame cropping end",   type = int)
    
    parser.add_argument("--start_length",  default = 8.5,  help = "Temporal start length", type = float)
    parser.add_argument("--frame_width",   default = 0.05, help = "Width of each frame", type = float)
    parser.add_argument("--sol",           default = 1.0,  help = "Speed of light", type = float)
    parser.add_argument("-l", "--load",    default = False, action = "store_true", help = "Whether to load from the cached histogram")
    parser.add_argument("-v", "--viz",     default = False, action = "store_true", help = "Visualize instead of saving image")
    parser.add_argument("--load_path1",    default = None, help = "Cached value path to be loaded from", type = str)
    parser.add_argument("--load_path2",    default = None, help = "Cached value path to be loaded from", type = str)

    if delayed_parse:
        return parser
    return parser.parse_args()

def smart_round(number, max_precision=2):
    precision = max_precision - len(str(int(number))) + (number < 0)
    return round(number, precision)

def single_normal_pdf(xs, mean, std):
    return 1 / std / np.sqrt(2 * np.pi) * np.exp(-0.5 * ((xs - mean) / std) ** 2)

def make_ticks(opts, interval = 1, num_ticks = 6):
    start_time = opts.start_length + opts.crop_start * opts.frame_width
    tick_pos = np.arange(num_ticks, dtype = np.float32) * interval + start_time
    tick_label = tick_pos / opts.sol
    return tick_pos, np.round(tick_label, decimals = 2)

def get_all_pdf(start, end, num_samples, mean1, mean2, std):
    xs_values = np.linspace(start, end, num_samples)
    pdf1 = single_normal_pdf(xs_values, mean1, std)
    pdf2 = single_normal_pdf(xs_values, mean2, std)
    return xs_values, 1/3 * pdf1 + 2/3 * pdf2

hist_labels = ('Vanilla sampling', 'Proposed sampling')
colors = ('#DC8686', '#7C93C3', '#363062')
alphas = (0.9, 0.9, 0.85)

if __name__ == "__main__":
    opts = get_tdom_options()
    
    sns.set_style('whitegrid')
    font_path = '../utils/font/biolinum.ttf'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    # Set the loaded font as the global font
    plt.rcParams['font.family'] = prop.get_name()
    plt.subplots_adjust(*(0.1, 0.111, 0.90, 0.95))
    plt.rc('font', size=14)       
    plt.rc('axes', labelsize=15)  
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14) 
    plt.rc('legend', fontsize=15) 
    plt.rc('figure', titlesize=12)
    
    for i in range(2):
        to_load = False
        if opts.load:
            load_path = getattr(opts, f'load_path{i + 1}')
            if load_path is None or not os.path.exists(load_path):
                CONSOLE.log(f"Path '{load_path}' not found.")
            else:
                to_load = True

        if to_load:
            CONSOLE.log(f"Cached values loaded from '{load_path}'.")
            xs, values = np.load(load_path)
        else:
            CONSOLE.log(f"Computing relative histogram: ts = '{opts.start_length}', frame = {opts.frame_width}, sol = {opts.sol}")
            input_path = getattr(opts, f'input_path{i + 1}')
            input_name = getattr(opts, f'input_name{i + 1}')
            all_trans = read_transient(input_path, input_name, 
                                   opts.num_transient, identity = False, extension = opts.ext)
            values    = all_trans.sum(axis = (-1, -2, -3))
            xs = opts.start_length + np.arange(values.shape[0], dtype = all_trans.dtype) * opts.frame_width
            xs /= opts.sol

        if opts.cache is not None:
            sol_print = 1 if abs(opts.sol - 1) < 1e-3 < 1000 else 'c'
            cache_name = f"{opts.cache}-{smart_round(opts.start_length)}-{smart_round(opts.frame_width)}-{sol_print}-{i}.npy"
            if not os.path.exists("./cached/"):
                os.makedirs("./cached/")
            caching = np.stack([xs, values], axis = 0)
            np.save(f"./cached/{cache_name}", caching)
            CONSOLE.log(f"Cache file saved to './cached/{cache_name}'")
        if opts.crop_start > 0 or opts.crop_end > 0:
            ratio = values.sum() / values[opts.crop_start:-opts.crop_end].sum()
            xs     = xs[opts.crop_start:-opts.crop_end]
            values = (values[opts.crop_start:-opts.crop_end] * ratio).astype(int)

        plt.bar(x = xs, height = values, width = opts.frame_width, color = colors[i], linewidth = 0.5, alpha = alphas[i], label = hist_labels[i])

    lines, labels = plt.gca().get_legend_handles_labels()

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0,0))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.ylabel('relative path sample count')
    plt.xlabel('full path time / ns')
    
    ax2 = plt.gca().twinx()
    xvalues, pdf_values = get_all_pdf(11.5, 16.5, 5000, 12.5, 16, 0.09)
    pdf_values /= pdf_values.max()
    ax2.plot(xvalues, pdf_values, color = colors[-1], alpha = alphas[-1], label='Weight', linestyle = '--')
    ax2.set_ylabel('Sensor temporal response weight')
    ax2.grid(visible = False)
    ax2.set_ylim(bottom=0)
    
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines += lines2
    labels += labels2
    
    plt.legend(lines, labels, loc='best')
    plt.xticks(*make_ticks(opts))
    
    if opts.viz:
        plt.show()
    else:
        plt.savefig("./output.png", dpi = 400)
