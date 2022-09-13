import argparse
import os

import cv2
from tqdm import tqdm

import make_util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make simulated dataset')
    parser.add_argument('--root_path', type=str, default='/mldisk/nfs_shared_/VCDB_simulated/simulated_frame_db')
    args = parser.parse_args()
    print(args)

    logo_img = cv2.imread('/workspace/imgs/default_logo.jpg')
    pip_img = cv2.imread('/workspace/imgs/default_pip.jpg')

    img_names = os.listdir(os.path.join(args.root_path, 'origin'))
    bar = tqdm(img_names, ncols=120, desc='make simulated dataset', unit='image', unit_scale=1)
    for img_name in bar:
        img = cv2.imread(os.path.join(args.root_path, 'origin', img_name))

        target = os.path.join(args.root_path, 'multi_BrC', img_name)
        t_img = make_util.crop(img, 0.5)
        t_img = make_util.brightness(t_img, -18)
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        cv2.imwrite(target, t_img)
