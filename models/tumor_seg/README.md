# Brain Tumor Segmentation
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/Jeetu95)
[![forthebadge](https://forthebadge.com/images/badges/check-it-out.svg)](https://github.com/Jeetu95/Brain-Tumor-Segmentation)
[![forthebadge](https://forthebadge.com/images/badges/uses-badges.svg)](https://forthebadge.com)<br>
[![HitCount](http://hits.dwyl.io/Jeetu95/Brain-Tumor-Segmentation.svg)](http://hits.dwyl.io/Jeetu95/Brain-Tumor-Segmentation)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Jeetu95/Brain-Tumor-Segmentation/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/Jeetu95/Brain-Tumor-Segmentation/issues)

This project uses [U-Net Architecture](https://arxiv.org/abs/1505.04597) to create segmentation masks for brain tumor images.

## Overview
- [Dataset Used](#Dataset-Used)
- [Data Augmentation](#Data-Augmentation)
- [Model Architecture](#Model-Architecture)
- [Training Process](#Training-Process)

### Dataset Used
Dataset used in this project was provided by Jun Cheng.<br>
This dataset contains 3064 T1-weighted contrast-enhanced images with three kinds of brain tumor. For a detailed information about the dataset please refer to this [site](https://figshare.com/articles/brain_tumor_dataset/1512427).
Version 5 of this dataset is used in this project. Each image is of dimension ```512 x 512 x 1``` , these are black and white images thus having a single channel.<br>
Some Data Samples<br>

Original Image             |  Mask Image
:-------------------------:|:-------------------------:
![](images/README/dataset_example.png)  |  ![](images/README/dataset_example_mask.png)


### Data Augmentation
The basic forms of data augmentation are used here to diversify the training data.
All the augmentation methods are used from [Pytorch's](https://pytorch.org) [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html) module.
- [Horizontally Flip](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.functional.hflip)
- [Vertically Flip](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.functional.vflip)
- [Rotation](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.functional.rotate) Between 75°-15°

Code Responsible for augmentation<br>
![Augmentation Code](images/README/data_aug.svg)<br>
Each augmentation method has a probability of 0.5 and the order of application is also random. For Rotation Augmentation the degree of rotation is chosen randomly between 75°-15°.


### Model Architecture
The model architecture is depicted in this picture.
![Model Architecture](images/README/architecture.png)


### Training Process
The model was trained on a [Nvidia GTX 1050Ti](https://www.geforce.com/hardware/desktop-gpus/geforce-gtx-1050-ti/specifications) 4GB GPU. Total time taken for model training was 6 hours and 45 minutes. We started with an initial learning rate of 1e-3 and reduced it by 85% on plateauing, final learning rate at the end of 100 epochs was 2.7249e-4.<br>
Some graphs indicating Learning Rate & Loss Value over 100 epochs are given below.

![LR Graph](images/README/lr_graph.png)
Learning Rate Graph in Tensorboard.<br>

![Loss Graph](images/README/loss_graph.png)
Loss Graph Plotted in [Matplotlib](https://matplotlib.org)<br>
![Loss Graph](images/README/loss_graph_2.png)
Loss Graph Plotted in Tensorboard<br><br>

To see the complete output produced during the training process check [this](logs/05-47-51_PM_on_May_20,_2019/training_output_log.txt)

## Installation
This project uses python3.

Clone the project.
```bash
git clone https://github.com/Jeetu95/Brain-Tumor-Segmentation.git
```
Install Pytorch from this [link](https://pytorch.org/get-started/locally/)<br>
Use pip to install all the dependencies
```bash
pip install -r requirements.txt
```
To open the notebook
```bash
jupyter lab
```
To see logs in Tensorboard
```bash
tensorboard --logdir logs --samples_per_plugin images=100
```
To setup the project dataset
```bash
python setup_scripts/download_dataset.py
python setup_scripts/unzip_dataset.py
python setup_scripts/extract_images.py
```

## Usage

```bash
python api.py --file <file_name> --ofp <optional_output_file_path>
python api.py --folder <folder_name> --odp <optional_output_folder_path>
python api.py -h
```

```
Available api.py Flags
--file  : A single image file name.
--ofp   : An optional folder location only where the output image will be stored. Used only with --file tag.

--folder: Path of a folder containing many image files.
--odp   : An optional folder location only where the output images will be stored. Used only with --folder tag.

-h, --help : Shows the help contents.
```
Some results generated by API are [here](images/API)

## Results
The mean [Dice Score](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) our model gained was 0.74461 in testing dataset of 600 images.<br>
From this we can conclude that in our testing dataset our constructed mask has a similarity of about 74% with the original mask.<br>
Some samples from our training dataset output are below. The top best results are [here](images).To see all the results click on this [Google Drive link](https://drive.google.com/drive/folders/1vwwUipaH9Yb0NLelv3lW-04E6WnVJ3nh?usp=sharing)<br>

.             |  .
:-------------------------:|:-------------------------:
![](images/0.98010_423.png)  |  ![](images/0.97981_1172.png)
![](images/0.97746_537.png)  |  ![](images/0.97623_636.png)
![](images/0.97441_1247.png)  |  ![](images/0.97391_373.png)
![](images/0.97316_425.png)  |  ![](images/0.97224_1400.png)
![](images/0.97216_631.png)  |  ![](images/0.97097_50.png)
![](images/0.97050_1465.png)  |  ![](images/0.96925_581.png)
![](images/0.96848_390.png)  |  ![](images/0.96812_222.png)
![](images/0.96669_14.png)  |  ![](images/0.96664_189.png)
![](images/0.96626_408.png)  |  ![](images/0.96605_994.png)
![](images/0.96603_170.png)  |  ![](images/0.96600_63.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT License](LICENSE)
