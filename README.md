# FLLENet
A Fused Low-Light Enhancement Framework for Advanced Lane Detection in Nighttime Driving Scenarios
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

