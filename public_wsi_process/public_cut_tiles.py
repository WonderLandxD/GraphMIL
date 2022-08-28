import glob
import h5py
import openslide
import os
from PIL import Image
import tqdm
import argparse
import numpy as np
from matplotlib import pyplot as plt


def cut_tiles(slide_path, h5_path, save_h5_to_tiles_path, slide_id):
    wsi = openslide.open_slide(slide_path)
    with h5py.File(h5_path, 'r') as hdf5_file:
        patch_level = hdf5_file['coords'].attrs['patch_level']
        patch_size = hdf5_file['coords'].attrs['patch_size']
        for idx in tqdm.tqdm(range(len(hdf5_file['coords']))):
        # for idx in range(len(hdf5_file['coords'])):
            coord = hdf5_file['coords'][idx]
            img = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
            img_RGB = img.convert('RGB')
            img_np = np.array(img_RGB)
            img_name = '{}_{}_{}_{}.png'.format(slide_id, patch_level, coord[0], coord[1])
            save_path = os.path.join(save_h5_to_tiles_path, img_name)
            plt.imsave(save_path, img_np)



parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--slide_dir', type=str, #default='D:/LocalPC_PythonProject/openslide_demo/img/tif/slides'
                    )
parser.add_argument('--h5_dir', type=str, #default='D:/LocalPC_PythonProject/openslide_demo/img/tif/h5'
                    )
parser.add_argument('--save_dir', type=str, #default='D:/LocalPC_PythonProject/openslide_demo/img/tif'
                    )
parser.add_argument('--sys_select', type=str, default=None, help='--win, windows system; --linux, linux system')
args = parser.parse_args()

if __name__ == '__main__':

    args = parser.parse_args()
    slide_dir = args.slide_dir
    save_dir = os.path.join(args.save_dir, 'tiles')
    h5_dir = args.h5_dir
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir, exist_ok=True)
    slide_path_list = glob.glob(os.path.join(slide_dir, '*.tif'))
    i = 1
    for slide_path in slide_path_list:
        if args.sys_select == 'win':
            slide_id = slide_path.split('\\')[-1].split('.')[0]
        elif args.sys_select == 'linux':
            slide_id = slide_path.split('/')[-1].split('.')[0]
        h5_path = os.path.join(h5_dir, '{}.h5'.format(slide_id))
        save_h5_to_tiles_path = os.path.join(save_dir, slide_id)
        print('progress: {:.2%}, {}/{} \t processing {}'.format(i/len(slide_path_list), i, len(slide_path_list),
                                                                 slide_id))
        if os.path.exists(save_h5_to_tiles_path):
            print('{} already exist in destination location, skipped'.format(slide_id))
            i += 1
            continue
        else:
            os.mkdir(save_h5_to_tiles_path)
            i += 1
            cut_tiles(slide_path=slide_path, h5_path=h5_path, save_h5_to_tiles_path=save_h5_to_tiles_path, slide_id=slide_id)
            print('{} is finished'.format(slide_id))




