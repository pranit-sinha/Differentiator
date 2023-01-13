# Differentiator
This is the repository for a differentiator, thie job of this model is to classify a given input histopathology patch as a callSpace Patch or a whiteSpace Patch. The model uses metdata generated from an image, such as individual channel mean, sum, and standrd deviations as well as total image metrics. We use this as an input to a decision tree classifier. The model is trained on 224x224px patches of histopathology retrieved and annotated by experts from IISER Pune, and is validated using a secondary unseen dataset of 625x625px images.


## Results
### Training
The model achieved a training time accuracy of 100% over unseen validation data and 100% accuracy over the entire training set. The confusion matrix and classification report can be seen below:


### Testing
The model was tested against 625x625px images. These images were resized to 224x224 pixels so account for real world downsampling and were fed into the model for testing. The model was able to achieve an accuracy of 95.27% on these test images. The confusion matrix and classification report can be found below:
