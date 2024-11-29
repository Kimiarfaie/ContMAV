# Open-World Semantic Segmentation for Ship Hull Inspection

This repository is adapted from the original [Open-World Semantic Segmentation Including Class Similarity](https://github.com/PRBonn/ContMAV.git), which was presented at IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR) 2024. The code has been customized and extended for the application of semantic segmentation in underwater ship hull inspection. Specifically improvements have been made to the "Feature loss" to adress some issues it has.

You can find the paper [here](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/sodano2024cvpr.pdf).

## Installation

Install the libraries of the `requirements.yml`, or create a conda environment by `conda env create -f requirements.yml` and then `conda activate openworld`.

The weights of ResNet34 with NonBottleneck 1D block pretrained on ImageNet are available [here](https://drive.google.com/drive/folders/1goULJjHp5-M7nUGlC52uvWaQxn2j3Za1?usp=sharing).

## Datasets

Information can be found regarding the datsets under src/datasets. 

## Training

You can choose your favourite hyperparameters configuration in `args.py`. For training, run
`python train.py --id <your_id> --dataset_dir <your_data_dir> --num_classes <N> --batch_size 8`.

The expected data structure is taken from Cityscapes. BDDAnomaly has been converted to Cityscapes format.

## Testing

Use the following command to evaluate your trained model:.

`python test.py --dataset 'liaci' --dataset_dir 'path-tp-dataset' --checkpoint_path 'path-to-checkpoint' --code_mode 'test_ow' --test_notes 'test_liaci_owthresh3' --batch_size 16 --num_classes <N>`.


