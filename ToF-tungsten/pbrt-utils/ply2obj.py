""" Convert ply file to obj file (then it will be converted to Tungsten wo3 file)
"""

import os
import tqdm
from plyfile import PlyData
from argparse import ArgumentParser

def get_configs():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_folder', type = str, required = True)
    parser.add_argument('-o', '--output_folder', type = str, required = True)
    return parser.parse_args()

def single_convert(ply_filename, obj_filename):
    convert(PlyData.read(ply_filename), obj_filename)

def convert(ply, obj):
    """ Modified from https://gist.github.com/randomize/a95e2db97b1277bbb49d4f2f59c08f3d """

    with open(obj, 'w') as f:
        f.write("# OBJ file\n")

        verteces = ply['vertex']
        
        for v in verteces:
            p = [v['x'], v['y'], v['z']]
            a = p
            f.write("v %.6f %.6f %.6f\n" % tuple(a) )

        try:
            for v in verteces:
                n = [ v['nx'], v['ny'], v['nz'] ]
                f.write("vn %.6f %.6f %.6f\n" % tuple(n))
        except ValueError:
            print(f"File '{obj[:-4]}.ply' has no vertex normal field.")
            pass
        
        if 'u' in verteces._property_lookup and 'v' in verteces._property_lookup:
            for v in verteces:
                t = [ v['u'], v['v'] ]
                f.write("vt %.6f %.6f\n" % tuple(t))
        elif 's' in verteces._property_lookup and 't' in verteces._property_lookup:
            for v in verteces:
                t = [ v['s'], v['t'] ]
                f.write("vt %.6f %.6f\n" % tuple(t))
        else:
            print(f"No UV coordinates found in {obj[:-4]}.ply")
            for v in verteces:
                f.write("vt 0.0 0.0\n")

        if 'face' in ply:
            for i in ply['face']['vertex_indices']:
                f.write("f")
                for j in range(i.size):
                    ii = [ i[j]+1, i[j]+1, i[j]+1 ]
                    f.write(" %d/%d/%d" % tuple(ii) )
                f.write("\n")
                
def traverse_folder_convert(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    all_file_name = os.listdir(input_folder)
    for file in tqdm.tqdm(all_file_name):
        if not file.endswith(".ply"): continue
        if "/" in file:
            name = file.split("/")[-1]
        else:
            name = file
        name_no_ext = name[:-4] 
        input_path = os.path.join(input_folder, name)
        output_path = os.path.join(output_folder, f"{name_no_ext}.obj")
        convert(PlyData.read(input_path), output_path)

if __name__ == "__main__":
    opts = get_configs()
    traverse_folder_convert(opts.input_folder, opts.output_folder)
    # single_convert("./staircase/mesh244.ply", "./staircase/mesh244.obj")