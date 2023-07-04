# GraphMIL
GraphMIL Research (Updating)

## I. UPDATE

***Updating on 2022.08.28***

- Used feature_vig_backbone.py as the GraphMIL backbone

- Used ResNet50 as the feature extractor (will update soon)

***Updating on 2022.08.29***

- Updated feature extractor on PUBLIC Datasets (win & linux version, based on *Openslide*), the process will upload later.

***Updating on 2022.08.30***

- Updated *wsi_process folder*, which contains *wsi_core folder*, *create_patches_fp.py*, *cut_tiles.py*, *feature_extractor.py*,

***Updating on 2022.09.04***

- The framework of GraphMIL has been improved, and back-propagation can now be performed (the depth and parameter improvement of the backbone still needs to be done in the *feature_vig_backbone.py* file, and the external model interface will be provided later)

- Updated *train.py* file, Model training with *train.py* is now available

***Updating on 2022.09.05***

- Updated training code of 4-fold cross-validation for the SYSFL dataset (PRIVATE DATASET)
- SYSFL WSIs Directory: `A:\Datasets\Histopathology\WSI_SYSFL (Local PC)`
- SYSFL features Directory: `/data_public/Wonderland/Datasets/Private/SYS_ThreeClasses_Features`

***Updating on 2022.09.12***

- Updated training code of 4-fold cross-validation for the Camelyon16 dataset (OPEN DATASET)
- Camelyon16 WSIs Directory: `/data_public/Wonderland/Datasets/Public/CAMELYON16/CAMELYON16`
- Camelyon16 features Directory: `/data_public/Wonderland/Datasets/Public/CAMELYON16/Cam16_20x_224/*/*/pt_files`

## II. PROCESS

### Step 1：Install Sdpc Library (If using Public Datasets, such as .svs, .tiff, skip this step)

`pip install sdpc-win` for windows
`pip install sdpc-linux` for linux

### Step 2: Create h5 Files (which contains the coordinates of tiles)

``` shell
python ./wsi_process/create_patches_fp.py --img_format IMAGE_FORMAT (os: .svs, .tif; sdpc: .sdpc)  --source WSI_DIRECTORY --step_size 224 --patch_size 224 --save_dir RESULTS_DIRECTORY --patch_level 1 
```

```bash
WSI_DIRECTORY/
	├── slide_1.svs (slide_1.sdpc)
	├── slide_2.svs (slide_2.sdpc)
	└── ...
```

```bash
RESULTS_DIRECTORY/
	├── masks
    		├── slide_1.jpg
    		├── slide_2.jpg
    		└── ...
	├── h5
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.jpg
    		├── slide_2.jpg
    		└── ...
	└── process_list_autogen.csv
```
Note that 
- if the IMAGE_FORMAT is .sdpc, h5 files will be stored on the WSI_DIRECTORY folder and stitches can not show. (These bugs will fix later, before this you need to drag all h5 files into the h5 folder manually)
- On windows system, the directory will be written like 'xx\xx\xx\xx\', but on linux system, the directory will be written the same as normal.

### Step 3: Get Tiles (patches) 

``` shell
python ./wsi_process/cut_tiles.py --slide_dir WSI_DIRECTORY  --h5_dir H5_DIRECTORY --save_dir TILES_DIRECTORY --sys_select SYSTEM --img_format IMAGE_FORMAT (os: .svs, .tif; sdpc: .sdpc)
```

```bash
TILES_DIRECTORY/
	├── slide_1 
		   ├── slide_1_tile_1.png
		   ├── slide_1_tile_2.png
		   └── ...

	├── slide_2 
		   ├── slide_2_tile_1.png
		   ├── slide_2_tile_2.png
		   └── ...
	├── ...
	└── ...
```

### Step 4: Get Feature Embeddings (based on ResNet50 Backbone, others will upload later)

``` shell
python ./wsi_process/feature_extractor.py --gpu GPU (single or multiple)  --tiles_dir TILES_DIRECTORY --feat_dir FEAT_DIRECTORY --batch_size BATCH_SIZE --feat_backbone FEATURE_BACKBONE
```

```bash
FEAT_DIRECTORY/
  ├── slide_1.pt
  ├── slide_2.pt
  ├── slide_3.pt 
  ├── ...
  └── ...
```

### Step 5: Training with GraphMIL

``` shell
python train.py --gpu 0 or MULTIPLE GPUS  --num_epoch NUM_EPOCH --lr LEARNING RATE --dataset SYS or Cam16 --upper_bound 8000 
```


*H&G Pathology AI Research Team*
