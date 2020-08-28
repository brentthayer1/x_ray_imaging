
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
	title="dist1" width="330" height="220" />

<img src="/images/plots/validation_images.png" alt="dist2"
	title="dist2" width="330" height="220" />  

When initially training a network to identify the two classes, the model was predicting pneumonia across the board.  

## Initial Model Metrics

<img src="/images/plots/fixed_roc_roc.png" alt="fix_roc_roc"
	title="fix_roc_roc" width="440" height="220" /> 

<img src="/images/plots/fixed_roc.png" alt="fix_roc"
	title="fix_roc" width="440" height="220" /> 

## Swish Vs RelU

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
 
## Final Model

<img src="/images/plots/ADD_MODEL_PATH.png" alt="architecture"
	title="architecture" width="440" height="550" /> 

|             | Test        |  Validation |
| :---------: | :---------: | :----------:|
| Loss        | 0.0000       | 0.0000     |
| Accuracy    | 0.0000       | 0.0000     |
| Precision   | 0.0000       | 0.0000     |
| Recall      | 0.0000       | 0.0000     |
| AUC         | 0.0000       | 0.0000     |
| True Pos    | 0.0000       | 0.0000     |
| True Neg    | 0.0000       | 0.0000     |
| False Pos   | 0.0000       | 0.0000     |
| False Neg   | 0.0000       | 0.0000     |


## Conclusion
Though I can not personally differentiate between a chest X-ray showing pneumonia and one that does not, the network seemed to pick up on the difference fairly quickly.  I never reached an accuracy rating over 0.92, but this model seems to be very effective.

## Slide Deck
[Please feel free to view my presentation slide deck with more charts and exciting things.](https://github.com/brentthayer1/x_ray_imaging/tree/master/presentation)