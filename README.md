# Enhanced Skip Net (ESNet) Model

Implimentation of ESNet using Pytorch framework.

Official paper link: [https://www.frontiersin.org/articles/10.3389/fpls.2019.01404/full](https://www.frontiersin.org/articles/10.3389/fpls.2019.01404/full "https://www.frontiersin.org/articles/10.3389/fpls.2019.01404/full")

![inferences](https://user-images.githubusercontent.com/39106620/114217881-0afe8400-9969-11eb-9e22-f1242b33f71f.gif)

Figure: Some predictions after retraining the ESNet model on 24 images for 15 epochs

## Get Started

### Environment

* Python 3.7 +
* PyTorch 1.8.0+cu101
* torchvision 0.8.1+cu101
* NumPy, SciPy, PIL
* tqdm 4.59.0 +
* cv2 4.3 +

## Get From Drive 

### [Dataset to train The Weights](https://drive.google.com/drive/folders/1Kjm1JbZkHjbbGurMD5pcw7IudjrAoCdD?usp=sharing) 
### [Dataset to retrain The Model](https://drive.google.com/drive/folders/10VaIId4WzPYqpJeVpo1F6kyGkTzs0a0p?usp=sharing)

### [Pre-trained Weights](https://drive.google.com/file/d/1TqmS0uPRU-9xubKZ4jYsaVTPBmFnOfSm/view?usp=sharing) 

### [Trained Model](https://drive.google.com/file/d/1hrRGjg2ScLUwIRmnZBpMry4TOp0_7_x_/view?usp=sharing) 

## Usage

### Train The ESNet from Scratch

* Make sure to replace dataset directories :
TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR
* Set the LOAD_MODEL to False.
*  Create a Directory "saved_images" in same location as train.py file, It's where the model will place the visualization of ground-truth and prediction after each Epoch.
* The number of Epochs must be 100 at least.
* Run train.py

### Train The ESNet from Pre-trained weights

* Make sure to replace dataset directories :
TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR
* Set the LOAD_MODEL to True.
* Add the saved weights "weights.pth.tar" in same loaction as train.py file.
*  Create a Directory "saved_images" in same location as train.py file, It's where the model will place the visualization of ground-truth and prediction after each Epoch.
* The number of Epochs can be 10 at least.
* Run train.py

### Make some Inferences

* Make sure to replace DATASET_Path with images path, and RESULTS_DIR with the directory where to save the inferences.
* Add the model "model.pth.tar" in same loaction as inference.py file.
* Run inference.py

