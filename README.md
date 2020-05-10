# Depth Map Prediction
This repository contains pytorch implementation of two reseach work in the area of depth map prediction. This is an unofficial implementation. For more details please refer to the paper.

"**Depth Map Prediction from a Single Image using a Multi-Scale Deep Network**" David Eigen, Christian Puhrsch, Rob Fergus. (2014) [[Paper Link]](https://arxiv.org/pdf/1406.2283.pdf)
```
@article{DBLP:journals/corr/EigenPF14,
  author    = {David Eigen and
               Christian Puhrsch and
               Rob Fergus},
  title     = {Depth Map Prediction from a Single Image using a Multi-Scale Deep
               Network},
  journal   = {CoRR},
  volume    = {abs/1406.2283},
  year      = {2014},
  url       = {http://arxiv.org/abs/1406.2283},
  archivePrefix = {arXiv},
  eprint    = {1406.2283},
  timestamp = {Mon, 13 Aug 2018 16:47:10 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/EigenPF14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
"**Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture**" David Eigen, Rob Fergus. (2015) [[Paper Link]](https://arxiv.org/pdf/1411.4734v4.pdf)
```
@article{DBLP:journals/corr/EigenF14,
  author    = {David Eigen and
               Rob Fergus},
  title     = {Predicting Depth, Surface Normals and Semantic Labels with a Common
               Multi-Scale Convolutional Architecture},
  journal   = {CoRR},
  volume    = {abs/1411.4734},
  year      = {2014},
  url       = {http://arxiv.org/abs/1411.4734},
  archivePrefix = {arXiv},
  eprint    = {1411.4734},
  timestamp = {Mon, 13 Aug 2018 16:47:14 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/EigenF14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Dataset
### NYU Depth Dataset V2 [[Dataset Webpage]](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
There are two modes present for the NYU Depth Dataset V2.
#### Light
This methods has divided labeled dataset into 2 parts as training and testing. approximately 10% images of each scene has been included in the test data for the uniformity. The purpose of this Light mode is for fast hypertuning or running the code in less spacious environment. 
* No of Train image: 1291
* No of Test image: 158


#### Raw
The training data is prepared using the scripts from the [repository](https://github.com/cogaplex-bts/bts). The test data of this mode is created from official train/test split of the NYU Depth dataset, so it can be used for reporting the accuracies. 
* No of Train image: 24231
* No of Test image: 654



#### KITTI Dataset [[Dataset Webpage]](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php)

## Architecture

## How to use the repository
```bash
# prepare the dataset
python scripts/nyu_depth_v2.py --mode raw
```

## results

## Pretrained Models
