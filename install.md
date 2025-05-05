# Installation

We provide the instructions to install the dependency packages.

## Requirements

We test the code in the following environments, other versions may also be compatible:

- CUDA 11.7
- Python 3.10
- Pytorch 2.0.1



## Setup

First, clone the repository locally.

```
git clone https://github.com/Linnea51/ZoomVG.git
```

Then, install Pytorch 2.0.1 using the conda environment.
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install the necessary packages and pycocotools.

```
pip install -r requirements.txt 
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Finally, compile CUDA operators.

```
cd models/ops
python setup.py build install
cd ../..
```