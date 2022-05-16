# VGG-Image-classification

**Introduction:**
The pre-trained model is a saved network that was trained on large dataset. In transfer learning, one can use these learned feature maps to avoid training from scratch. We have used VGG 16 and VGG 19 models as pre-trained model for classification.

**Dataset**: The dataset used for image classification contains images of building, forest, street, mountain, glacier, and sea. 

There are two ways by which transfer learning is implemented:

•	**Feature Extraction**: The pre-trained model is used as a base model to extract features from the new data samples. The classification layers of the pre-trained model are specific to the task, hence replaced by layers according to new task. This way we do not have to train whole model from scratch, only the last classification layers are trained on new dataset. Using previous network for feature extraction, we are freezing all the top layers of the model.

•	**Fine Tuning**: To improve performance further, we can unfreeze some layers from the frozen model. This helps to learn high-level features specific to the new data samples. In this case, along with classification layers unfreeze layers are also trained on new dataset. This increases number of training parameters.

**Steps:**

•	To use pre-trained model as a base model, set **include_top= False**, in order to exclude model’s fully connected layers.
   ``` 
    conv_base = VGG19(weights='imagenet',
                    include_top= False,
                    input_shape=input_shape)
   ```       
   
•	For feature extraction, freeze entire model by setting **model.trainable= False**.

•	For fine tuning, some layers are unfrozen by specifying those number of layers.
```
for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
```

•	Build classification layers specific to the task and train the model.


