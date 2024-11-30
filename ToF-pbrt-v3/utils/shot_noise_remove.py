import os
import sys
import imageio
import numpy as np
import taichi as ti
from pathlib import Path
from taichi.math import vec3

@ti.kernel
def filtering(src: ti.template(), dst: ti.template(), threshold: float):
    # This filtering should be improved.
    for i, j in dst:
        center_pix = src[i + 1, j + 1]
        valid = False
        pix_val_sum = vec3([0, 0, 0])
        for k_x in range(3):
            for k_y in range(3):
                if k_x == 1 and k_y == 1: continue
                pix = src[i + k_x, j + k_y]
                norm = (pix - center_pix).norm()
                if norm < threshold:    # if the current pixel has similiar adjacent pixel: meaning that the pixel is not shot noise
                    valid = True
                    break
                pix_val_sum += pix
        if valid:
            dst[i, j] = center_pix
        else:
            dst[i, j] = pix_val_sum / 8.
            
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ./shot_noise_remove.py <path-to-image> <filtering shot noise threshold>")
    ti.init(arch = ti.cpu)
    image = imageio.imread(sys.argv[1]).astype(np.float32)
    h, w, _ = image.shape
    img_field = ti.Vector.field(3, float, (h + 2, w + 2))
    out_field = ti.Vector.field(3, float, (h, w))
    pad_img = np.pad(image, ((1, 1), (1, 1), (0, 0)))
    img_field.from_numpy(pad_img)
    filtering(img_field, out_field, float(sys.argv[2]))
    image = out_field.to_numpy()
    path = Path(sys.argv[1])
    parent = str(path.parent)
    stem   = str(path.stem)
    imageio.imwrite(os.path.join(parent, f"{stem}-filtered.png"), image.astype(np.uint8))