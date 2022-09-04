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

- Updated 4-fold cross-validation for the SYSFL dataset

## II. PROCESS

### Step 1：Install Sdpc Library (If using Public Datasets, such as .svs, .tiff, skip this step）

`pip install sdpc-win` for windows system; `pip install sdpc-linux` for linux system

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
python train.py --gpu 0 or MULTIPLE GPUS  --num_epoch NUM_EPOCH --lr LEARNING RATE --dataset SYS (Currently only the SYS dataset) --upper_bound 8000 
```

## III. RESEARCH

### NO.1 --- 乳腺癌er pr her2指标 by FYQ & ZZ

1. **调研乳腺癌er pr her2指标相关文献**，包括两大类。一，针对该医学问题，提出的利用计算机辅助诊断的全自动系统的文章(例如雨秋提到的几篇nature文献)；二，基于此全自动系统，提出的深度学习方法(例如MIL方法：Attention-Based MIL，CLAM，TransMIL)。注意：两项调研对象并不分离。建议将已调研的文献总结下来，记录成表格或文字。
2.  基于已调研的文献，利用深度学习中的MIL方法，完成针对乳腺癌er, pr, her2指标在WSI级别上的**全自动化**指标评定任务。（弱监督学习，只需WSI级别标签，无需patch标注）
3. 与医生**商量课题方案**，具体包括：
- *针对此医学问题的相关研究现状*
- *自己目前已完成的任务*
- *仍需改进的地方和想要达到的目标结果*
- *可发表的文章刊物*
4. (Optional) 利用GraphMIL的Backbone进行实验，调整模型深度、模块等架构，记录相关信息，整理相关结果。（GraphMIL的模型代码以及实验结果会持续更新在github上，方便参考）

### NO.2 --- 前列腺分级 by ZQL & ZBQ
1. **调研前列腺分级的相关文献**，包括两大类。一，针对该医学问题，提出的利用计算机辅助诊断的全自动系统的文章(例如之前和冰倩、骐来讨论的Teacher-Student模型文章）；二，基于此全自动系统，提出的深度学习方法(例如MIL方法：Attention-Based MIL，CLAM，TransMIL)。注意：两项调研对象并不分离。建议将已调研的文献总结下来，记录成表格或文字。
2. **整理常见的几种MIL方法**（TransMIL论文中的Baseline对比试验），做到后续的代码只需更改main函数的模型参数即可，且结果保存在不同文件夹下。（便于日后整理记录，以及熟悉代码的整理过程；若每种MIL方法涉及到的特征提取器不一致，统一采用ImageNet预训练的ResNet50进行，若有直接端到端：从patch图像预测WSI结果的，可分开考虑）--- 雨秋和珍珍可以利用这里写好的代码进行自己的MIL方法实验
3. 与何老师沟通目前进展，并**与医生商量课题方案**，具体包括：
- *针对此医学问题的相关研究现状*
- *自己目前已完成的任务*
- *仍需改进的地方和想要达到的目标结果*
- *可发表的文章刊物*
4. (Optional) 利用GraphMIL的Backbone进行实验，调整模型深度、模块等架构，记录相关信息，整理相关结果。（GraphMIL的模型代码以及实验结果我会持续更新在github上，方便大家参考）

**·注：两人之间要明确知晓、熟悉所研究内容。同时分工也须明确，切忌一个人干完所有的活；阶段工作完成后，相互给对方两人介绍，完成对接**

### NO.3 --- *GraphMIL：Multiple Instance Learning Based on Graph* by ZZ & ZBQ & FYQ & ZQL & LJW（目前由ljw完善代码，并持续更新）

1. **整理之前实现的HE通道图的前置任务代码和结果，实现基于HE通道图的前置任务**包括：一，HE图像生成H、E通道图的代码；二，前置任务的代码（将Encoder部分添加backbone选择项-ResNet18，ResNet34，ResNet50，EfficientNet-b0，EfficientNet-b1，SwinTransformer-Tiny）做到后续的代码实验只需调整main函数的model参数，且结果保存在相应模型下的文件夹即可（在此期间可以学习相关模型的文献和代码部分，可利用timm包等帮助实现模型部署）；三，整理好前置任务的结果，包括生成出的H、E通道图和模型解码出的H、E通道图（在此期间逐步上传之前切出来的patch，并利用这些patch进行通道处理。进度可能会比较长，作为长期工作进行）；以及各个Encoder通过前置任务保存下来的权重（注意：保存时记得将模型各超参数也一并记录下来，方便后续整理规划）
2. **处理H, E通道图放入self-attention模块的任务**
3. **考虑ROI Selector**
4. **添加Position Embedding**
5. **考虑如何处理每张WSI的tiles数量上界问题**
6. **考虑tiles预测值聚合至WSI预测值的问题**

Good Luck & Have Fun !

*H&G Pathology AI Research Team*
