import sys
import imageio
import numpy as np

if __name__ == "__main__":
    image_file      = sys.argv[1]
    output_img_file = sys.argv[2]
    quantile        = 0.99
    
    patch_pos  = [(1121, 165), (849, 542), (787, 196), (992, 250), (735, 284), (436, 117), (506, 189), (1027, 196), (1071, 364), (922, 335), (740, 364), (746, 357), (747, 366), (725, 356)]
    patch_size = [(3, 1), (2, 1), (2, 1), (1, 2), (2, 2), (2, 2), (1, 2), (2, 2), (2, 2), (1, 2), (2, 1), (2, 2), (2, 2), (3, 2)]
    margin = 1
    
    image = np.load(image_file)
    
    for (x, y), (dx, dy) in zip(patch_pos, patch_size):
        margin_patch = image[y - margin:y + dy + margin, x - margin:x + dx + margin]
        patch = image[y:y + dy, x:x + dx]
        rim_pixel = (dx + 2 * margin) * 2 + dy * 2
        color = (margin_patch.sum(axis = (0, 1)) - patch.sum(axis = (0, 1))) / rim_pixel
        image[y:y + dy, x:x + dx] = color

    np.save(image_file, image)    
    
    qnt = np.quantile(image, quantile)
    imageio.imwrite(output_img_file, ((image / qnt).clip(0, 1) * 255).astype(np.uint8))
    