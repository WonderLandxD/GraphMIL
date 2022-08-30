# GraphMIL
GraphMIL Research (Updating)

***Updating on 2022.08.28***

- Using feature_vig_backbone.py as the GraphMIL backbone

- Using ResNet50 as the feature extractor (will update soon)

***Updating on 2022.08.29***

- Update feature extractor on PUBLIC Datasets (win & linux version, based on *Openslide*), the process will upload later.

***Updating on 2022.08.30***

- Update *wsi_process folder*, which contains *wsi_core folder*, *create_patches_fp.py*, *cut_tiles.py*, *feature_extractor.py*,

### Step 1：Install Sdpc Library

`pip install sdpc-win` for windows system

` ` for linux system (will upload later)

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


## Research One --- 乳腺癌er pr her2指标 by FYQ & ZZ
1. 利用MIL方法（常见的几种方法均可，参考TransMIL论文中的Baseline对比试验，可自行选择一种），完成WSI级别上的全自动化指标评定任务（弱监督学习，只需WSI级别标签，无需patch标注）
2. 调研乳腺癌er pr her2指标相关文献，包括两大类。一，相关医学问题的已研究进展；二，针对此医学问题的相关方法。注意：两项调研对象并不分离。建议将已调研的文献总结下来，记录成表格或文字。
3. 与医生商量课题方案，具体包括：目前已完成的任务、该课题可发表的文章刊物、针对文章发表的标准还需改进的地方。（中山附一的组会下周应该会开，请提前做好准备）
4. 整理之前实现的HE通道图的前置任务代码和结果。包括：一，HE图像生成H、E通道图的代码；二，前置任务的代码（将Encoder部分添加backbone选择项-ResNet18，ResNet34，ResNet50，EfficientNet-b0，EfficientNet-b1，SwinTransformer-Tiny）做到后续的代码实验只需调整main函数的model参数，且结果保存在相应模型下的文件夹即可（在此期间可以学习相关模型的文献和代码部分，可利用timm包等帮助实现模型部署）；三，整理好前置任务的结果，包括生成出的H、E通道图和模型解码出的H、E通道图（在此期间逐步上传之前切出来的patch，并利用这些patch进行通道处理。进度可能会比较长，作为长期工作进行）；以及各个Encoder通过前置任务保存下来的权重（注意：保存时记得将模型各超参数也一并记录下来，方便后续整理规划）
5. 利用GraphMIL的Backbone进行实验，调整模型深度、模块等架构，记录相关信息，整理相关结果。（GraphMIL的模型代码以及实验结果我会持续更新在github上，方便大家参考）

## Research Two --- 前列腺分级 by ZQL & ZBQ
1. 整理常见的几种MIL方法（TransMIL论文中的Baseline对比试验），做到后续的代码只需更改main函数的参数即可，且结果保存在不同文件夹下。（便于日后整理记录，以及熟悉代码的整理过程；若每种MIL方法涉及到的特征提取器不一致，统一采用ImageNet预训练的ResNet50进行，若有直接端到端：从patch图像预测WSI结果的，可分开考虑）
2. 调研前列腺分级的相关文献，包括两大类。一，相关医学问题的已研究进展；二，针对此医学问题的相关方法。注意：两项调研对象并不分离。建议将已调研的文献总结下来，记录成表格或文字。
3. 与何老师沟通目前进展，并与医生商量课题方案，具体包括：目前已完成的任务、该课题可发表的文章刊物、针对文章发表的标准还需改进的地方。（决定与老师和医生商量前可先跟ljw说）
4. 利用GraphMIL的Backbone进行实验，调整模型深度、模块等架构，记录相关信息，整理相关结果。（GraphMIL的模型代码以及实验结果我会持续更新在github上，方便大家参考）

**·注：两人之间要明确知晓、熟悉所研究内容。同时分工也须明确，切忌一个人干完所有的活；阶段工作完成后，相互给对方两人介绍，完成对接**

## Research Three --- *GraphMIL：Multiple Instance Learning Based on Graph* by ZZ & ZBQ & FYQ & ZQL & LJW（目前由ljw完善代码，并持续更新）

Good Luck & Have Fun !

*H&G Pathology AI Research Team*

