import os
import configargparse
from rich.console import Console
CONSOLE = Console(width = 128)

command_prefix = "python ./zoom_in.py --iname "
iname = [
    "../build/vpt/12hours-baseline/cornell-vpt_0016.exr",
    "../build/vpt/2048-baseline/cornell-vpt_0016.exr",
    "../build/vpt/12hours-baseline/cornell-vpt_0008.exr",
    "../build/vpt/2048-tsample/cornell-vpt_0016.exr",
]

def get_options():
    # IO parameters
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config',  
                                     is_config_file=True, help='Config file path')
    parser.add_argument("--folder",        default = None,            type = str, help = "The folder to be listed, all the folders in this folder will be enumerated for zoom_in.py")
    parser.add_argument("--output_folder", default = "./time-gated/", type = str, help = "The folder to be listed, all the folders in this folder will be enumerated for zoom_in.py")
    parser.add_argument("--oname",         default = "gated-crop",    type = str, help = "The folder to be listed, all the folders in this folder will be enumerated for zoom_in.py")
    parser.add_argument("--pattern",       default = [], nargs="*",   type = str, help = "Pattern that should be found in 'folder' names, the folders with all patterns can be used")
    parser.add_argument("--npattern",      default = [], nargs="*",   type = str, help = "Pattern that can't be found in 'folder' names, the folders with all patterns can't be used")
    parser.add_argument("--file_name",     default = "result",        type = str, help = "The name of the file for each folder. A common name is assumed for all folders")
    parser.add_argument("--json",          default = None,            type = str, help = "If we want to use a saved json file, we can invoke the program with --json <filename> only")
    parser.add_argument("--qnt",           default = 0.99,            type = float, help = "The name of the file for each folder. A common name is assumed for all folders")
    parser.add_argument("--flip_pfm",      default = False,           action = 'store_true', help = 'Whether to horizontally flip the PFM format file.')

    return parser.parse_args()


if __name__ == "__main__":
    opts = get_options()
    if opts.json is not None:
        cmd = f"python ./zoom_in.py --input_json {opts.json} --output_folder {opts.output_folder} --oname {opts.oname} --qnt {opts.qnt}"      
    else:
        if opts.folder is not None and opts.file_name is not None:
            CONSOLE.log(f"Overriding <iname> using the provided folder and filename.")
            iname.clear()
            folders = os.listdir(opts.folder)
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
                        if pattern in folder: 
                            continue_flag = True
                            break
                    if continue_flag: continue
                for file in (f"{opts.file_name}_0000", f"{opts.file_name}_0", f"{opts.file_name}"):
                    for ext in ('npy', 'exr', 'pfm'):
                        fp = os.path.join(opts.folder, folder, f"{file}.{ext}")
                        if os.path.exists(fp):
                            iname.append(fp)
                            break
                    else: continue
                    break
                else:
                    CONSOLE.log(f"Warning: for folder \'{folder}\', there is no file with name [yellow] \'{opts.file_name}\'.")
        for file in iname:
            command_prefix = f"{command_prefix}{file} "
        cmd = f"{command_prefix}--output_folder {opts.output_folder} --oname {opts.oname} --qnt {opts.qnt}"     
    if opts.flip_pfm:
        cmd=f"{cmd} --flip_pfm"   
    os.system(cmd)