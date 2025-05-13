"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_valid(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # 验证图片文件是否完整
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image: {e}")
        return False

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, check=False):
    images = []
    images_rel = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if check:
                    is_valid = is_image_valid(path)
                else:
                    is_valid = True
                if is_valid:
                    path_rel = os.path.relpath(path, start=dir)
                    images.append(path)
                    images_rel.append(path_rel)
                else:
                    continue
    return images, images_rel
