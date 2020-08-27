
![title_image]("/Volumes/b/Galvanize/DS-RFT4/capstones-RFT4/x_ray_imaging/images/title_image.png")

# X-Ray Imaging 

## Topic
Use a convolutional neural network to determine if a chest x-ray shows signs of COVID, pneumonia, or are normal lungs.

## Data
https://www.kaggle.com/alexkort/xray-covid19

## Process
This dataset had a pretty substantial class imbalance.  When initially training a network to identify three classes, the model was more or less just predicting one class across the board.  After weighting classes differently, I decided to remove the COVID X-rays and focus on a binary classification model, trying to predict the precense of pneumonia in an X-Ray.

The was a large range of image sizes in the data set.  I resized all of the images to the same size, and scaled the pixels.  The majority of the images had the lungs centered, but there were a handful that were not centered and had a very large black boundary.  Even though the majority of the images are grayscale, there were a few oddball colored X-rays.  I pushed the whole dataset to grayscale.  This was the extent of image preprocessing that took place. 
