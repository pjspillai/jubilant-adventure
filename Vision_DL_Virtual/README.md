#### README.md

##### Create and Setup up virtual Anaconda environment for DL, Vision

```
conda create -y -n visionResearch python=3.5.5
source activate visionResearch
```


### Install Conda Packages 
* OpenCV
* Caffe2
* PyTorch
* CNTK

`
sh all_requirements.sh
`

### Install python pip packages
##### Data Science , Utilities
* matplotlib
* numpy
* scipy
* scikit
* Pillow
* pandas
* milk
* reques
* h5py
* tqdm
* yaml
* protobuf

##### Deep Learning
* Keras
* Tensorflow
* Torch
* Microsoft CNTK Toolkit 
* MxNet (Cuda 8.0)

`pip install --yes -r pip_requirements.txt`


### Final Check

`python3`

###### Inside Python Console
```
import cv2
import caffe2
import tensorflow
import mxnet
import keras
import torch
import numpy as np
import scipy
```


###### Delete Environment

`conda env remove -n visionResearch`


