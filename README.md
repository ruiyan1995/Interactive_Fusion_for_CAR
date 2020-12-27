# Interactive_Fusion_for_CAR
Pytorch implementation of [Interactive Fusion of Multi-level Features for Compositional Activity Recognition](https://github.com/ruiyan1995/Interactive_Fusion_for_CAR).

## Get started
### Prerequisite
Our approach is tested on only Ubuntu with GPU and it needs at least 16G GPU memory. The neccseearay packages can be install by following commonds:
```
conda create -n Interactive_Fusion python=3.6
conda activate Interactive_Fusion
pip install pyyaml matplotlib tensorboardx opencv-python
pip install torch torchvision
```
### Preprocess datasets
#### Something-Else
- Download [Something-Something Dataset](https://github.com/joaanna/something_else) and [Something-Else Annotations](https://github.com/joaanna/something_else).
- Extract (or softlink) videos under ```dataset/sth_else/videos```, and then dump the frames into ```dataset/sth_else/frames``` by the following commands:
```
bash tools/dump_frames_sth.sh
```

#### Charades
- Download [Charades Dataset](https://prior.allenai.org/projects/charades) (scaled to 480p) and [Action Genome Annotations](https://github.com/JingweiJ/ActionGenome).
- Extract (or softlink) videos under ```dataset/charades/videos```, put the annotations into ```dataset/charades/annotations```, and then dump the frames into ```dataset/charades/frames``` by the following commands:
```
bash tools/dump_frames_char.sh
```


#### Get frame information 
i.e., height, width, and the number of frames in each video
```
bash tools/get_frame_info.sh 'sth_else'
bash tools/get_frame_info.sh 'charades'
```


### Train a Standard Model from Scratch
```
# Compositional setting for Something-Else
python main.py --cfg STHELSE_COM
# Compositional setting for Charades
python main.py --cfg CHARADES_COM
```

## Citation
If you wish to refer to the results of this work, please use the following BibTeX entry.
```
@article{yan2020interactive,
  title={Interactive Fusion of Multi-level Features for Compositional Activity Recognition},
  author={Yan, Rui and Xie, Lingxi and Shu, Xiangbo and Tang, Jinhui},
  journal={arXiv preprint arXiv:2012.05689},
  year={2020}
}
```
## Acknowledgments
Our code is built on the Pytorch implementation of [STIN](https://github.com/joaanna/something_else) proposed by joaanna.

## Contact Information
Feel free to create a pull request or contact me by Email = ["ruiyan", at, "njust", dot, "edu", dot, "cn"], if you find any bugs.