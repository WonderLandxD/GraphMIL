import os
from public_wsi_process.wsi_core import WholeSlideImage


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
				  patch_size = 256, step_size = 256,
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8},
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False,
				  seg = False, save_mask = True,
				  stitch= False,
				  patch = False, auto_skip=True, process_list = None):

    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]

    df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path)

        w, h = WSI_object.level_dim[current_seg_params]


        current_vis_params = {}
        current_filter_params = {}
        current_seg_params = {}
        current_patch_params = {}







if __name__ == '__main__':
    args = parser.parse_args()

    coord_save_dir = os.path.join(args.save_dir, 'coords_h5')
    PatchImg_save_dir = os.path.join(args.save_dir, 'PatchImgs')
    mask_save_dir = os.path.join(args.save_dir, 'contours')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    print('source: ', args.source)
    print('coord_save_dir: ', coord_save_dir)
    print('PatchImg_save_dir: ', PatchImg_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)

    while True:
        value = input('Continue? [y/n]')
        if value == 'y':
            break
        elif value == 'n':
            ValueError('CANCEL! Some dirs may not be correct.')

    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'coord_save_dir': coord_save_dir,
                   'PatchImg_save_dir': PatchImg_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    coord_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'coord_params': coord_params,
                  'vis_params': vis_params}

    print(parameters)

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                           patch_size=args.patch_size, step_size=args.step_size,
                                           seg=args.seg, use_default_params=False, save_mask=True,
                                           stitch=args.stitch,
                                           patch_level=args.patch_level, patch=args.patch,
                                           process_list=process_list, auto_skip=args.no_auto_skip)










