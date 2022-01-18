# Optical Character Recognition (OCR)

**Michiel Jacobs**  
Master of Chemistry: Chemical Theory, (Bio)Molecular Design and Synthesis  
Faculty of Sciences and Bio-Engineering Sciences  

___

## How to do inference

1. Setup the conda environment with: `conda env create -f environment.yml`
2. Activate the environment: `conda activate dl-proj`
3. Run `inference.py`. By default, it will make predictions on the handwriting of me and some of my collegues.
4. To use your own image, change the `IMAGEPATH` variable to were you have stored the image. The only requirement is that the image should be a square. (Let's say as captured by some object detection.)

## Project overview

* The `experimental` folder contains some preliminary code for exploration of the dataset and a basic model.
* The `production` folder contains the code to train the model, training was done on Hydra for speed.
* The `inference` folder contains the code to make predictions from.

## Sources

The code for calculating the accuracy was inspired on [the official tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#test-the-network-on-the-test-data).  
LeNet-5 from the paper: LeCun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. Gradient-Based Learning Applied to Document Recognition. Proc. IEEE 1998, 86 (11), 2278–2323. https://doi.org/10.1109/5.726791.