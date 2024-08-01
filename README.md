# FLLENet
A Robust and Real-time Lane Detection Method in Low-Light Scenarios to Advanced Driver Assistance Systems
# NightLane
链接：[https://pan.baidu.com/s/13DQjCsVDATCFC9g6Lx8reA ](https://pan.baidu.com/s/1G6PvB1altFal1MmK0WYJ8g?pwd=j5ft )
提取码：:j5ft

## Installation
<!--
Please refer to [INSTALL.md](INSTALL.md) for installation.
-->

### Clone this repository
```
git clone https://github.com/turoad/lanedet.git
```
We call this directory as `$LANEDET_ROOT`

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n lanedet python=3.8 -y
conda activate lanedet
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.

conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.1 -c pytorch

# Or you can install via pip
pip install torch==1.8.0 torchvision==0.9.0

# Install python packages
python setup.py build develop
```

### Data preparation

#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.
## Getting Started

### Training

For training, run

```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_ids]
```


For example, run
```Shell
python main.py configs/resa/resa50_culane.py --gpus 0
```

### Testing
For testing, run
```Shell
python main.py [configs/path_to_your_config] --validate --load_from [path_to_your_model] [gpu_num]
```

For example, run
```Shell
python main.py configs/resa/resa50_culane.py --validate --load_from culane_resnet50.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.

For example, run
```Shell
python main.py configs/resa/resa50_culane.py --validate --load_from culane_resnet50.pth --gpus 0 --view
```

### Inference
See `tools/detect.py` for detailed information.
```
python tools/detect.py --help

usage: detect.py [-h] [--img IMG] [--show] [--savedir SAVEDIR]
                 [--load_from LOAD_FROM]
                 config

positional arguments:
  config                The path of config file

optional arguments:
  -h, --help            show this help message and exit
  --img IMG             The path of the img (img file or img_folder), for
                        example: data/*.png
  --show                Whether to show the image
  --savedir SAVEDIR     The root of save directory
  --load_from LOAD_FROM
                        The path of model
```
To run inference on example images in `./images` and save the visualization images in `vis` folder:
```
python tools/detect.py configs/resa/resa34_culane.py --img images\
          --load_from resa_r34_culane.pth --savedir ./vis
```
## Acknowledgement
<!--ts-->
* [Li-Chongyi/Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)
* [XingangPan/SCNN](https://github.com/XingangPan/SCNN)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->

<!-- 
## Citation
If you use
```
@misc{peng2024FLLENet,
  author =       { },
  title =        {A Robust and Real-time Lane Detection Method in Low-Light Scenarios toAdvanced Driver Assistance Systems},
  howpublished = {}},
  year =         {2024}
}
``` -->


