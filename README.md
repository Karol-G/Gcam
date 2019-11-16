# cnn_interpretability

## What does this repo contain?

* The folder kazuto1011-grad-cam contains a working GradCAM version to play/test with.
* The folder 1Konny-grad-cam++ contains a working GradCAM++ version to play/test with.


## How to use kazuto1011-grad-cam?

* Create a conda environment by importing the environment.yml with: **conda env create -f environment.yml**
* Run GradCAM with: **python main.py demo1 -a vgg19 -t features.35 -i samples/cat_dog.png -k 3 --cuda**
* Parameters are explained in kazuto1011's repo
* Repo: https://github.com/kazuto1011/grad-cam-pytorch

## How to use 1Konny-grad-cam++?

* Create a conda environment by importing the environment.yml with: **conda env create -f environment.yml**
* Run the main.py
* Repo: https://github.com/1Konny/gradcam_plus_plus-pytorch
* Original implementation: https://github.com/adityac94/Grad_CAM_plus_plus