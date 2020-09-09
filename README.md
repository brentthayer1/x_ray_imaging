
![](/images/title_image.png)

# X-Ray Imaging 

## Topic
Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.

Can a convolutional neural network determine if a chest x-ray shows signs of pneumonia?  I hope to train a model that is about 90% accurate when predicting on if an x-ray shows signs of pneumonia.

## Data
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

The dataset included images for a pneumonia class and a normal class.

### Pneumonia
<img src="/images/xrays/pneum/person49_virus_101.jpeg" alt="pneumonia"
	title="pneumonia" width="250" height="250" />

### Normal
<img src="/images/xrays/norm/NORMAL2-IM-0198-0001.jpeg" alt="normal"
	title="normal" width="250" height="250" />

## Process
This dataset had a pretty substantial class imbalance.  

<img src="/images/plots/training_images.png" alt="dist1"
	title="dist1" width="350" height="230" />

<img src="/images/plots/validation_images.png" alt="dist2"
	title="dist2" width="350" height="230" />  


## Initial Model 

<img src="/images/plots/swish_to_relu.png" alt="relu"
	title="relu" width="350" height="230" /> 


### Second Model

<img src="/images/plots/added_augmentation_60_epochs.png" alt="aug"
	title="aug" width="350" height="230" /> 
 
## Final Model

<img src="/images/plots/FINAL_MODEL_2.png" alt="final"
	title="final" width="400" height="200" /> 

<img src="/images/plots/FINAL_MODEL_2_ROC.png" alt="final_roc"
	title="final_roc" width="400" height="200" /> 

Accuracy: 0.9030   
Precision: 0.8723   
Recall: 0.9866 
AUC: 0.9631


## Conclusion
Though I can not personally differentiate between a chest X-ray showing pneumonia and one that does not, the network seemed to pick up on the difference fairly quickly.  I never reached an accuracy rating over 0.92, but this network seems to be very effective in correctly identifying pneumonia in an X-ray image.

## Slide Deck
[Presentation Slide Deck](https://github.com/brentthayer1/x_ray_imaging/tree/master/presentation)

