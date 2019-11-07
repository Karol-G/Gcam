# cnn_interpretability

## What does this repo contain?

* The folder kazuto1011-grad-cam-pytorch contains a working grad-cam version to play/test with.


## How to use kazuto1011-grad-cam-pytorch?

* Create a conda environment by importing the environment.yml with: **conda env create -f environment.yml**
* Run grad-cam with: **python main.py demo1 -a vgg19 -t features.35 -i samples/cat_dog.png -k 3 --cuda**
* Parameters are explained in kazuto1011's repo
* Repo: https://github.com/kazuto1011/grad-cam-pytorch