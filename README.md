# Open-World Semantic Segmentation Including Class Similarity

This is the code repository of the paper Open-World Semantic Segmentation Including Class Similarity, accepted to the IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR) 2024.

You can find the paper [here](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/sodano2024cvpr.pdf).

## Installation

Install the libraries of the `requirements.yml`, or create a conda environment by `conda env create -f requirements.yml` and then `conda activate openworld`.

The weights of ResNet34 with NonBottleneck 1D block pretrained on ImageNet are available [here](https://drive.google.com/drive/folders/1goULJjHp5-M7nUGlC52uvWaQxn2j3Za1?usp=sharing).

## Training

You can choose your favourite hyperparameters configuration in `args.py`. For training, run
`python train.py --id <your_id> --dataset_dir <your_data_dir> --num_classes <N> --batch_size 8`.

The expected data structure is taken from Cityscapes. BDDAnomaly has been converted to Cityscapes format.

## Testing

Use the following command to test the trained model on your dataset.

`python test.py --dataset 'liaci' --dataset_dir 'path-tp-dataset' --checkpoint_path 'path-to-checkpoint' --code_mode 'test_ow' --test_notes 'test_liaci_owthresh3' --batch_size 16 --num_classes <N>`.


