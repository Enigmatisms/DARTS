from ast import Str
import sys
import subprocess
import os
from enum import Enum
from argparse import ArgumentParser
from pathlib import Path
import copy

class Command(Enum):
    merge = 'merge'
    variance = 'variance'
    variancemap = 'variancemap'

    def __str__(self):
        return self.value

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('command', type=Command, choices=list(Command))
    parser.add_argument('image_dir', type=Path, default=None)
    parser.add_argument('image_name')
    parser.add_argument('num_images', type=int)
    parser.add_argument('--hdrmanip-path', type=Path, help='manually specify path to hdrmanip executable')
    parser.add_argument('--hdrbatch-path', type=Path, help='manually specify path to hdrbatch executable')

    return parser.parse_args()

def run_with_echo(args):
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        encoding='utf-8'
    )
    print(result.stdout, end='')
    print(result.stderr, end='')

def find_hdrmanip():
    ret = None
    possible_names=['hdrmanip.exe', 'hdrmanip']
    for name in possible_names:
        for path in Path('build').rglob(name):
            ret = path.relative_to('.')
            print('Info: found hdrmanip at: {}'.format(ret))
            break
        if ret is not None:
            break
    return ret

def find_hdrbatch():
    ret = None
    possible_names=['hdrbatch.exe', 'hdrbatch']
    possible_envs = ['HDRBATCH_PATH', 'HDRVIEW_PATH']
    for env in possible_envs:
        env_path = os.environ.get(env)
        if env_path is None:
            continue
        for name in possible_names:
            for path in Path(env_path).glob(name):
                ret = path
                print('Info: found hdrbatch at: {}'.format(ret))
                break
        if ret is not None:
            break
    return ret

def main():
    opts = parse_args()

    command = opts.command
    image_dir = opts.image_dir
    image_name_prototype = copy.deepcopy(opts.image_name)
    num_images = opts.num_images
    hdrmanip_path = opts.hdrmanip_path
    hdrbatch_path = opts.hdrbatch_path

    if hdrmanip_path is None:
        hdrmanip_path = find_hdrmanip()
        if hdrmanip_path is None:
            print('Error: cannot find hdrmanip. Please provide path manully using --hdrmanip-path')
            return -3

    if not os.path.exists(image_dir):
        print('Error: path not exist: {}'.format(image_dir))
        return -1

    image_paths = []
    for i in range(0, num_images):
        image_name = image_name_prototype.replace("[]", str(i))
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print("warning: file not exist: {}".format(image_path))

    output_dir = image_dir
    if command == Command.merge:
        output_path_hdr = os.path.join(output_dir, "merged_{}.pfm".format(num_images))
        output_path_ldr = os.path.join(output_dir, "merged_{}.png".format(num_images))

        args = [hdrmanip_path, "-o", output_path_hdr, "-m", "-f", "pfm"]
        args.extend(image_paths)
        run_with_echo(args)

        args = [hdrmanip_path, "-o", output_path_ldr, "-f", "png", output_path_hdr]
        run_with_echo(args)
    elif command == Command.variance:
        args = [hdrmanip_path, "--variance"]
        args.extend(image_paths)
        run_with_echo(args)
    elif command == Command.variancemap:
        if hdrbatch_path is None:
            hdrbatch_path = find_hdrbatch()
            if hdrbatch_path is None:
                print('Error: cannot find hdrbatch using envvar HDRBATCH_PATH, HDRVIEW_PATH. Please specify manually using --hdrbatch-path.')
                return -4

        out_name = "variance_map.pfm"
        out_path = os.path.join(output_dir, out_name)
        args = [hdrbatch_path, "--variance={}".format(out_path)]
        args.extend(image_paths)
        run_with_echo(args)
    else:
        print("Unknown command: {}".format(command))
        return -2

    return 0

if __name__ == "__main__":
    main()
