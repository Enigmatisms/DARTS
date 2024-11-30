import configargparse

def get_tdom_options(delayed_parse = False):
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",  is_config_file = True, help='Config file path')
    parser.add_argument("--input_path",    default = "", required = True, help = "Input simulation data folder", type = str)
    parser.add_argument("--input_name",    default = "foam", help = "Input simulation data name", type = str)
    parser.add_argument("--ext",           default = "auto", help = "Input file extension (auto-detection by default)", type = str)
    parser.add_argument("--num_transient", default = 500, help = "Number of transient image", type = int)
    
    parser.add_argument("--time_length",   default = -1.0, help = "Temporal duration", type = float)
    parser.add_argument("--sol",           default = 1.0, help = "Speed of light", type = float)
    parser.add_argument("--qnt",           default = 0.99, help = "Normalizing quantile", type = float)
    parser.add_argument("--window_mode",   default = "diag_side_mean", choices=['diag_tri', 'diag_side_mean', 'whole', 'manual'], help = "Window cropping mode", type = str)
    parser.add_argument("-s", "--skip_analysis", default = False, action = "store_true", help = "Whether to skip temporal analysis")
    parser.add_argument("-f", "--flip",    default = False, action = "store_true", help = "Whether to flip the input PFM file")
    parser.add_argument("--skip_output",   default = False, action = "store_true", help = "Whether to skip outputing images")
    parser.add_argument("--ratio_threshold", default = -1, help = "(% 0.99 quantile) as filtering threshold", type = float)
    
    parser.add_argument("--crop_x",         default = 256, help = "Cropping center x", type = int)
    parser.add_argument("--crop_y",         default = 265, help = "Cropping center y", type = int)
    parser.add_argument("--crop_rx",        default = 0, help = "Cropping radius x", type = int)
    parser.add_argument("--crop_ry",        default = 0, help = "Cropping radius y", type = int)

    if delayed_parse:
        return parser
    return parser.parse_args()

def get_viz_options(delayed_parse = False):
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",  is_config_file = True, help='Config file path')
    parser.add_argument("--input_path1",    default = "", required = True, help = "Input simulation data folder", type = str)
    parser.add_argument("--input_path2",    default = "", required = True, help = "Input simulation data folder", type = str)
    parser.add_argument("--input_name1",    default = "foam", help = "Input simulation data name", type = str)
    parser.add_argument("--input_name2",    default = "foam", help = "Input simulation data name", type = str)
    parser.add_argument("--crop_x",         default = 256, help = "Cropping center x", type = int)
    parser.add_argument("--crop_y",         default = 265, help = "Cropping center y", type = int)
    parser.add_argument("--crop_rx",        default = 16, help = "Cropping radius x", type = int)
    parser.add_argument("--crop_ry",        default = 16, help = "Cropping radius y", type = int)
    parser.add_argument("--num_transient",  default = 500, help = "Number of transient image", type = int)
    parser.add_argument("--ext",            default = "auto", help = "Input file extension (auto-detection by default)", type = str)
    
    parser.add_argument("--time_length",   default = -1.0, help = "Temporal duration", type = float)
    parser.add_argument("--sol",           default = 1.0, help = "Speed of light", type = float)
    parser.add_argument("--qnt",           default = 0.99, help = "Normalizing quantile", type = float)
    parser.add_argument("--window_mode",   default = "whole", choices=['diag_tri', 'diag_side_mean', 'whole'], help = "Window cropping mode", type = str)

    if delayed_parse:
        return parser
    return parser.parse_args()

def get_tran_comp_options(delayed_parse = False):
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",  is_config_file = True, help='Config file path')
    parser.add_argument("--input_dir",      default = "", required = True, help = "Input simulation data directory", type = str)
    parser.add_argument("--folders",        default = [], required = True, nargs = "*", help = "Input simulation data folders", type = str)
    parser.add_argument("--input_names",    default = [], required = True, nargs = "*", help = "Input simulation data file names", type = str)
    parser.add_argument("--legends",        default = [], nargs = "*", help = "Plottig labels", type = str)
    parser.add_argument("--crop_x",         default = 256, help = "Cropping center x", type = int)
    parser.add_argument("--crop_y",         default = 265, help = "Cropping center y", type = int)
    parser.add_argument("--crop_rx",        default = 16, help = "Cropping radius x", type = int)
    parser.add_argument("--crop_ry",        default = 16, help = "Cropping radius y", type = int)
    parser.add_argument("--num_transient",  default = 500, help = "Number of transient image", type = int)
    parser.add_argument("--ext",            default = "auto", help = "Input file extension (auto-detection by default)", type = str)
    
    parser.add_argument("--time_length",   default = -1.0, help = "Temporal duration", type = float)
    parser.add_argument("--sol",           default = 1.0, help = "Speed of light", type = float)
    parser.add_argument("--qnt",           default = 0.99, help = "Normalizing quantile", type = float)
    parser.add_argument("--window_mode",   default = "whole", choices=['whole'], help = "Window cropping mode (only 'whole' is allowed here)", type = str)

    if delayed_parse:
        return parser
    return parser.parse_args()

def get_img_opts(delayed_parse = False):
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config",  is_config_file = True, help='Config file path')
    parser.add_argument("--input_path1",   default = "", required = True, help = "Input simulation data folder", type = str)
    parser.add_argument("--input_path2",   default = "", required = True, help = "Input simulation data folder", type = str)
    parser.add_argument("--input_name1",   default = "0000", help = "Input simulation data name", type = str)
    parser.add_argument("--input_name2",   default = "0000", help = "Input simulation data name", type = str)
    parser.add_argument("--qnt",           default = 0.99, help = "Normalizing quantile", type = float)

    if delayed_parse:
        return parser
    return parser.parse_args()