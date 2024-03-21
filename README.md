# ELiTe: Efficient Image-to-LiDAR Knowledge Transfer

This repository is for **ELiTe** introduced in the following paper:

Zhibo Zhang, Ximing Yang, Weizhong Zhang and Cheng Jin, "ELiTe: Efficient Image-to-LiDAR Knowledge Transfer for Semantic
Segmentation," *2024 IEEE International Conference on Multimedia and Expo (ICME)*, 2024.

## Installation

### Requirements

```shell
# for ELiTe
conda create -n elite python=3.9 -y
conda activate elite
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install pytorch_lightning==1.5.0 torchmetrics==0.5
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install spconv-cu113==2.3.6
pip install easydict
pip install setuptools==59.5.0

# for AdaLoRA
git clone https://github.com/QingruZhang/AdaLoRA.git
cd AdaLoRA/
pip install -e loralib/

# for SAM-PLG
pip install opencv-python
```

## Data Preparation

### SemanticKITTI

Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and additionally
the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip) from
the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract everything into the same
folder.

```
./dataset/
├── 
├── ...
└── SemanticKitti/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |   |   ├── 000000.label
        |   |   ├── 000001.label
        |   |   └── ...
        |   └── image_2/ 
        |   |   ├── 000000.png
        |   |   ├── 000001.png
        |   |   └── ...
        |   calib.txt
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

## Training

You can run the training with

```shell
python main.py --log_dir elite
```

The output will be written to `logs/SemanticKITTI/elite` by default.

## Testing

You can run the testing with

```shell
python main.py --test --num_vote 12 --checkpoint logs/SemanticKITTI/elite/version_0/checkpoints/last.ckpt
```

Here, `num_vote` is the number of views for the test-time-augmentation (TTA). `num_vote=1` denotes there is no TTA used.

## Pseudo-Label Generation

```shell
python amg.py
```

## Acknowledgements

Code is built based on [2DPASS](https://github.com/yanx27/2DPASS), [Segment Anything](https://github.com/facebookresearch/segment-anything), and [AdaLoRA](https://github.com/QingruZhang/AdaLoRA).

## License

This repository is released under MIT License (see LICENSE file for details).



