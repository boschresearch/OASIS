# You Only Need Adversarial Supervision for Semantic Image Synthesis

Official PyTorch implementation of the ICLR 2021 paper "You Only Need Adversarial Supervision for Semantic Image Synthesis". The code allows the users to
reproduce and extend the results reported in the study. Please cite the paper when reporting, reproducing or extending the results.

[[OpenReview](https://openreview.net/forum?id=yvQKLaqNE6M)]  [[Arxiv](https://arxiv.org/abs/2012.04781)]  

# Overview 

This repository implements the OASIS model, which generates realistic looking images from semantic label maps. In addition, many different images can be generated from any given label map by simply resampling a noise vector (first two rows of the figure below). The model also allows to just resample parts of the image (see the last two rows of the figure below). Check out the paper for details, as well as the appendix, which contains many additional examples.


<p align="center">
<img src="overview.png" >
</p>



## Setup
First, clone this repository:
```
git clone https://github.com/boschresearch/OASIS.git
cd OASIS
```

The code is tested for 3.7.6 and the packages listed in [oasis.yml](oasis.yml). 
The basic requirements are PyTorch and Torchvision. 
The easiest way to get going is to install the oasis conda environment via 
```
conda env create --file oasis.yml
source activate oasis
```
## Datasets 

For COCO-Stuff, Cityscapes or ADE20K, please follow the instructions for the dataset preparation as outlined in [https://github.com/NVlabs/SPADE](https://github.com/NVlabs/SPADE).

## Training the model

To train the model, execute the training scripts in the ```scripts``` folder. However, in these scripts you first need to specify the path to the data folder. Via the ```--name``` paramter the experiment can be given a unique identifier. The experimental results are then saved in the folder ```./checkpoints```, where a new folder for each run is created with the specified experiment name. You can also specify another folder for the checkpoints using the ```--checkpoints_dir``` parameter.
If you want to continue training, start the respective script with the ```--continue_train``` flag.

## Testing the model

To test a model, execute the testing scripts in the ```scripts``` folder. The ```--name``` parameter should correspond to the experiment name that you want to test, and the ```--checkpoints_dir``` should the folder where the experiment is saved (default: ```./checkpoints```). These scripts will generate images from a pretrained model.

## Pretrained models

We will upload the pretrained models soon.

## Citation
If you use this work please cite
```
@inproceedings{schonfeld_sushko,
  title={You Only Need Adversarial Supervision for Semantic Image Synthesis},
  author={Sch{\"o}nfeld, Edgar and Sushko, Vadim and Zhang, Dan and Gall, Juergen and Schiele, Bernt and Khoreva, Anna},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}   
```

## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Contact
Please feel free to contact us personally if you have questions, need help, or need explanations. 
Write to one of the following email addresses:

edgarschoenfeld@live.de  
vad221@gmail.com  
edgar.schoenfeld@bosch.com  
vadim.sushko@bosch.com  

