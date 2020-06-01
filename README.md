# Gcam (Grad-Cam)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-available-blue.svg)](https://karol-g.github.io/Gcam)
[![Pypi](https://img.shields.io/badge/pypi-v0.0.6-blue.svg)](https://pypi.org/project/gcam/)
![Python package](https://github.com/Karol-G/Gcam/workflows/Python%20package/badge.svg)

Gcam is an easy to use Pytorch framework that makes model predictions more interpretable for humans. 
It allows the generation of attention maps with multiple methods like Guided Backpropagation, 
Grad-Cam, Guided Grad-Cam and Grad-Cam++.

## Features

* Works with classification and segmentation data / models
* Works with 2D and 3D data
* Supports Guided Backpropagation, Grad-Cam, Guided Grad-Cam and Grad-Cam++
* Attention map evaluation with given ground truth masks
* Option for automatic layer selection

## Installation
You can install Gcam via pip with: <br/>
`pip install gcam`

## Documentation
Gcam is fully documented and you can view the documentation under: <br/> 
https://karol-g.github.io/Gcam

## Examples

|                                            |                #1 Classification (2D)                 |                  #2 Segmentation (2D)                 |                       #3 Segmentation (3D)            |
| :----------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: |
|                  Image                     |        ![](examples/images/class_2D_image.jpg)        |        ![](examples/images/seg_2D_image.jpg)          |        ![](examples/images/seg_3D_image.jpg)          |
|          Guided backpropagation            |        ![](examples/images/class_2D_gbp.jpg)          |        ![](examples/images/seg_2D_gbp.jpg)            |        ![](examples/images/seg_3D_gbp.jpg)            |
|                 Grad-Cam                   |        ![](examples/images/class_2D_gcam.jpg)         |        ![](examples/images/seg_2D_gcam.jpg)           |        ![](examples/images/seg_3D_gcam.jpg)           |
|              Guided Grad-Cam               |        ![](examples/images/class_2D_ggcam.jpg)        |        ![](examples/images/seg_2D_ggcam.jpg)          |        ![](examples/images/seg_3D_ggcam.jpg)          |
|               Grad-Cam++                   |        ![](examples/images/class_2D_gcampp.jpg)       |        ![](examples/images/seg_2D_gcampp.jpg)         |        ![](examples/images/seg_3D_gcampp.jpg)         |


## Install gcam from source

* Install Pytorch from https://pytorch.org/get-started/locally/
* Run `python setup.py sdist bdist_wheel` to create the gcam package in the `dist` directory
* Navigate to `dist` with `cd dist`
* Install gcam with `pip install gcam-XXX-py3-none-any.whl`

## Install gcam requirements

* Install Pytorch from https://pytorch.org/get-started/locally/
* Run `pip install -r requirements.txt`

## Usage

```python
# Import gcam
from gcam import gcam

# Init your model and dataloader
model = MyCNN()
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Inject model with gcam
model = gcam.inject(model, output_dir="attention_maps", save_maps=True)

# Continue to do what you're doing...
# In this case inference on some new data
model.eval()
for i, batch in enumerate(data_loader):
    # Every time forward is called, attention maps will be generated and saved in the directory "attention_maps"
    output = model(batch)
    # more of your code...
```

## Demo

You can find a Jupyter Notebook on how to use Gcam with the nnUNet for handeling 3D data inside the folder `demos`.
