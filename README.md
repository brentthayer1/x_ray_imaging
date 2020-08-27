
![](/images/title_image.png)

# X-Ray Imaging 

## Topic
Can a convolutional neural network determine if a chest x-ray shows signs of pneumonia?

## Data
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

The dataset included images for a pneumonia class and a normal class.

### Pneumonia
![](/images/xrays/pneum/person49_virus_101.jpeg)
### Normal
![](/images/xrays/norm/NORMAL2-IM-0198-0001.jpeg)

After browsing through all of the images quickly, I noticed that the majority of them have an 'R' in the top left corner.  There would be no need for arranging images based on posterior or anterior shots.  I made the assumption that all of these images were anterior.

In addition to this, I also noticed that the majority of these images were roughly the same size and the lungs were more or less centered in the frame.  There were a handful of exceptions, but I decided to see what the network would do with them.  
![](/images/xrays/oddball/person1706_bacteria_4516.jpeg)
![](/images/xrays/oddball/person1712_bacteria_4529.jpeg)

## Process
This dataset had a pretty substantial class imbalance.  
![](/images/plots/training_images.png)
![](/images/plots/validation_images.png)


When initially training a network to identify the two classes, the model was predicting pneumonia across the board.  

The was a large range of image sizes in the data set.  I resized all of the images to the same size, and scaled the pixels.  The majority of the images had the lungs centered, but there were a handful that were not centered and had a very large black boundary.  Even though the majority of the images are grayscale, there were a few oddball colored X-rays.  I pushed the whole dataset to grayscale.  This was the extent of image preprocessing that took place. 
