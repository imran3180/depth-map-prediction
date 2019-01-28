## Depth Map Prediction from a Single Image using a Multi-Scale Deep Network
  1. [depth-map-prediction](https://github.com/imran3180/depth-map-prediction)
  2. [unet-depth-prediction](https://github.com/DikshaMeghwal/unet-depth-prediction)
----
This repository is the first part of the project and Pytorch implementation of Depth Map Prediction from a Single Image using a Multi-Scale Deep Network by David Eigen, Christian Puhrsch and Rob Fergus. [Paper Link](https://cs.nyu.edu/~deigen/depth/depth_nips14.pdf)

<p align="center">
  <img src="https://s2.gifyu.com/images/output_Ky1KUn.gif" alt="https://gifyu.com/image/wZwF" alt="monodepth">
</p>



Architecture
----------
<p align="center">
  <img src="https://s2.gifyu.com/images/Screen-Shot-2019-01-26-at-6.39.35-PM.png" alt="https://gifyu.com/image/wZwY" alt="monodepth">
</p>

Data
----------
We used [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) as our dataset. We used Labeled dataset (~2.8 GB) of NYU Depth Dataset which provides 1449 densely labeled pairs of aligned RGB and depth images. We divided labeled dataset into three parts (Training - 1024, Validation - 224, Testing - 201) for our project. NYU Dataset also provides Raw dataset (~428 GB) on which we couldn't train due to machine capacity.

Training & Validation
-----------

Evaluation
-------------


Contributors
---------------------------------

- [Imran](https://github.com/imran3180/)
- [Diksha Meghwal](https://github.com/DikshaMeghwal/)

