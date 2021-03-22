# Interactive_Fusion_for_CAR
Pytorch implementation of [Interactive Fusion of Multi-level Features for Compositional Activity Recognition](https://arxiv.org/pdf/2012.05689).

## Get started
### Prerequisite
Our approach is tested on only Ubuntu with GPU and it needs at least 16G GPU memory. The neccseearay packages can be installed by the following commonds:
```
conda create -n Interactive_Fusion python=3.6
conda activate Interactive_Fusion
pip install pyyaml matplotlib tensorboardx opencv-python
pip install torch torchvision
```
### Preprocess datasets
#### Something-Else
- Download [Something-Something Dataset](https://github.com/joaanna/something_else) and [Something-Else Annotations](https://github.com/joaanna/something_else).
- Extract (or softlink) videos under ```dataset/sth_else/videos```, and then dump the frames into ```dataset/sth_else/frames``` by the following command:
```
bash tools/dump_frames_sth.sh
```

#### Charades
- Download [Charades Dataset](https://prior.allenai.org/projects/charades) (scaled to 480p) and [Action Genome Annotations](https://github.com/JingweiJ/ActionGenome).
- Extract (or softlink) videos under ```dataset/charades/videos```, put the annotations into ```dataset/charades/annotations```, and then dump the frames into ```dataset/charades/frames``` by the following command:
```
bash tools/dump_frames_char.sh
```


#### Get video information
Get some video information, such as the height and width of the video, and the number of frames in each video. Alternatively, you can also download ```video_info.json``` from [here](https://github.com/ruiyan1995/Interactive_Fusion_for_CAR#Pre-generated-files).
```
bash tools/get_video_info.sh 'sth_else'
bash tools/get_video_info.sh 'charades'
```

### Pre-generated files
You can check some necessary files in [Baidu Cloud](https://pan.baidu.com/s/18AAu3yg04rWEanJ_LVn3yg) (code: **gl0v**
), such as the annotations and video_info.json into ```dataset``` and the data-split settings in ```dataset_splits```. **Download and put them in the same path.**

### Train a Standard Model from Scratch
```
# Compositional setting for Something-Else
python main.py --cfg STHELSE/COM/GT/OURS
# Fewshot setting for Something-Else
python main.py --cfg STHELSE/FEWSHOT/GT/OURS/base
python main.py --cfg STHELSE/FEWSHOT/GT/OURS/5shot
python main.py --cfg STHELSE/FEWSHOT/GT/OURS/10shot
<!-- # Compositional setting for Charades
python main.py --cfg CHARADES_COM -->
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