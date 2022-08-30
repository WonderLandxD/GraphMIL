import argparse
import os
from sdpc.Sdpc import Sdpc
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Code to patch WSI using SDPC files')
parser.add_argument('--source', type=str, default=None, help='path to folder containing raw sdpc image files')
parser.add_argument('--sys_select', type=str, default=None, help='system selection --linux or win')
parser.add_argument('--patch_size', type=int, default=224)
parser.add_argument('--patch_level', type=int, default=1)
parser.add_argument('--save_dir', type=str, default=None, help='directory to save processed data')

if __name__ == '__main__':
    args = parser.parse_args()

    source_path = args.source
    patch_size = args.patch_size
    patch_level = args.patch_level
    save_dir = args.save_dir

    files = os.listdir(source_path)
    for idx, file in enumerate(files):
        print('file = ', file)
        slide_id, _ = file.split('.')
        fi_path = os.path.join(source_path, slide_id)

        print('progress : {:.2%}, {}/{} \t processing {}'.format(idx+1/len(files), idx+1, len(files), slide_id))

        if os.path.exists(fi_path):
            print('{} already exist in destination location, skipped'.format(slide_id))
            continue
        else:
            os.mkdir(fi_path)
            sdpc_path = os.path.join(source_path, file)
            wsi = Sdpc(sdpc_path)

            wsi_x, wsi_y = wsi.level_dimensions[0]

            for i in range(int(np.floor(wsi_x / patch_size))):
                for j in range(int(np.floor(wsi_y / patch_size))):
                    coord = [i * patch_size, j * patch_size]
                    img = wsi.read_region((coord[0], coord[1]), patch_level, (patch_size, patch_size))
                    img_RGB = img.convert('RGB')
                    img_np = np.array(img_RGB)
                    img_RGB_mean = np.mean(img_np[:, :, :])
                    img_RGB_var = np.var(img_np[:, :, :])

                    if img_RGB_mean < args.RGB_TH and img_RGB_var > args.RGB_var_TH:
                        img_name = '{}_{}_{}_{}.png'.format(slide_id, patch_level, coord[0], coord[1])
                        save_path = os.path.join(save_dir, slide_id, img_name)
                        plt.imsave(save_path, img_np)

                        print('{} is finished'.format(slide_id))





