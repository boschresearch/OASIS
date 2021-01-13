# You Only Need Adversarial Supervision for Semantic Image Synthesis

Official PyTorch implementation of the ICLR 2021 paper "You Only Need Adversarial Supervision for Semantic Image Synthesis". The code allows the users to
reproduce and extend the results reported in the study. Please cite the paper when reporting, reproducing or extending the results.

[[OpenReview](https://openreview.net/forum?id=yvQKLaqNE6M)]  [[Arxiv](https://arxiv.org/abs/2012.04781)]  

# Overview 

This repository implements the OASIS model, which generates realistic looking images from semantic label maps. In addition, many different images can be generated from any given label map by simply resampling a noise vector (first two rows of the figure below). The model also allows to just resample parts of the image (see the last two rows of the figure below). Check out the paper for details, as well as the appendix, which contains many additional examples.


<p align="center">
<img src="overview.png" >
</p>

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication [cited above | paper reference]. It will neither be
maintained nor monitored in any way.

## Setup

The code is tested for 3.7.6 and the packages listed in [oasis.yml](oasis.yml). 
The easiest way to get going is to install the oasis conda environment via 
```
conda env create --file oasis.yml
source activate oasis
```

## Training the model

## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
