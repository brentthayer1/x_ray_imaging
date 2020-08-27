
![](/images/title_image.png)

# X-Ray Imaging 

## Topic
Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.

Can a convolutional neural network determine if a chest x-ray shows signs of pneumonia?

## Data
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

The dataset included images for a pneumonia class and a normal class.

### Pneumonia
<img src="/images/xrays/pneum/person49_virus_101.jpeg" alt="pneumonia"
	title="pneumonia" width="250" height="250" />

### Normal
<img src="/images/xrays/norm/NORMAL2-IM-0198-0001.jpeg" alt="normal"
	title="normal" width="250" height="250" />

After browsing through all of the images quickly, I noticed that the majority of them have an 'R' in the top left corner.  There would be no need for arranging images based on posterior or anterior shots.  I made the assumption that all of these images were anterior.
In addition to this, I also noticed that the majority of these images were roughly the same size and the lungs were more or less centered in the frame.  There were a handful of exceptions, but I decided to see what the network would do with them.

<img src="/images/xrays/oddball/person1706_bacteria_4516.jpeg" alt="oddball1"
	title="oddball1" width="250" height="125" />

<img src="/images/xrays/oddball/person1712_bacteria_4529.jpeg" alt="oddball2"
	title="oddball2" width="250" height="250" />

## Process
This dataset had a pretty substantial class imbalance.  

<img src="/images/plots/training_images.png" alt="dist1"
	title="dist1" width="330" height="220" />

<img src="/images/plots/validation_images.png" alt="dist2"
	title="dist2" width="330" height="220" />  

When initially training a network to identify the two classes, the model was predicting pneumonia across the board.  

## Initial Model Metrics

<img src="/images/plots/fixed_roc_roc.png" alt="fix_roc_roc"
	title="fix_roc_roc" width="440" height="220" /> 

<img src="/images/plots/fixed_roc.png" alt="fix_roc"
	title="fix_roc" width="440" height="220" /> 

## Mid Way

### Swish Activations On All Layers

<img src="/images/plots/50_epochs_swish_roc.png" alt="swish_roc"
	title="swish_roc" width="440" height="220" /> 

<img src="/images/plots/50_epochs_swish.png" alt="swish"
	title="swish" width="440" height="220" />

### ReLU Activations On All Layers

<img src="/images/plots/swish_to_relu_roc.png" alt="relu_roc"
	title="relu_roc" width="440" height="220" /> 

<img src="/images/plots/swish_to_relu.png" alt="relu"
	title="relu" width="440" height="220" /> 

## Final Steps

### Added Image Augmentation Back In

<img src="/images/plots/added_augmentation_60_epochs_roc.png" alt="aug_roc"
	title="aug_roc" width="440" height="220" /> 

<img src="/images/plots/added_augmentation_60_epochs.png" alt="aug"
	title="aug" width="440" height="220" /> 

The was a large range of image sizes in the data set.  I resized all of the images to the same size, and scaled the pixels.  The majority of the images had the lungs centered, but there were a handful that were not centered and had a very large black boundary.  Even though the majority of the images are grayscale, there were a few oddball colored X-rays.  I pushed the whole dataset to grayscale.  This was the extent of image preprocessing that took place. 
