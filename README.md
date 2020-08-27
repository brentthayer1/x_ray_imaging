
![](/images/title_image.png)

# X-Ray Imaging 

## Topic
Use a convolutional neural network to determine if a chest x-ray shows signs of pneumonia.

## Data
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Process
This dataset had a pretty substantial class imbalance.  
![](/images/plots/training_images.png)
![](/images/plots/validation_images.png)


When initially training a network to identify the two classes, the model was predicting pneumonia across the board.  

The was a large range of image sizes in the data set.  I resized all of the images to the same size, and scaled the pixels.  The majority of the images had the lungs centered, but there were a handful that were not centered and had a very large black boundary.  Even though the majority of the images are grayscale, there were a few oddball colored X-rays.  I pushed the whole dataset to grayscale.  This was the extent of image preprocessing that took place. 
