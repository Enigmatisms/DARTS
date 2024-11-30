import sys
import struct
import numpy as np
from pathlib import Path

def read_pfm(filename, flip = False):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        if flip:
            return np.flip(np.flipud(np.reshape(decoded, shape)), axis = 1) * scale
        return np.flipud(np.reshape(decoded, shape)) * scale
    
def write_pfm(file, image, scale = 1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        print(f'Output image dtype must be float32, currently: {image.dtype.name}')
        print(f'Applying type casting.')
        image = image.astype(np.float32)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
    file.write(b'%f\n' % scale)
    image.tofile(file)  
    
def get_title_str(name):
    if "origin" in name:
        return "Vanilla PT"
    elif "darts" in name:
        if "point" in name:
            return "DARTS PP (Ours)"
        else:
            return "DARTS PT (Ours)"
    elif "pb" in name or "beam" in name:
        return "Photon Beams (1D)"
    else:
        return "Photon Points (2D)"
    