import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.models import load_model
import os
import cv2
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import kl_divergence
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.metrics import poisson
from tensorflow.keras.applications import EfficientNetB0


import numpy as np

data = "/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/DataSet"
datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.2)
# Training Data
train_generator = datagen_train.flow_from_directory(
        data,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset = 'training')
#Validation Data
valid_generator = datagen_train.flow_from_directory(
        data,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset = 'validation',
        shuffle=False)


effnet = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
        )
for layer in effnet.layers:
    layer.trainable = True
model = Sequential()
model.add(effnet)
model.add(layers.GlobalMaxPooling2D(name="gap"))

model.add(layers.Flatten(name="flatten"))
model.add(layers.Dropout(0.2, name="dropout_out"))
model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(2, activation='softmax', name="fc_out"))
model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
if not os.path.exists(f'/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/models'):
        os.makedirs(f'/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/models')
# Model Summary


#TF_CPP_MIN_LOG_LEVEL=2
# Training the model

print("------------------------------------------")
print(f'Training the model with {train_generator.samples} training samples and {valid_generator.samples} validation samples')
print("------------------------------------------")
# history = model.fit(train_generator, validation_data = valid_generator, epochs=40)

# # for layer in effnet.layers:
# #     if 'dense' in layer.name:
# #         layer.trainable = True
# #     if 'softmax' in layer.name:
# #         layer.trainable = True
# #     else:
# #         layer.trainable = False

# # model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])
# # history = model.fit(train_generator, validation_data = valid_generator, epochs=40)

# print("------------------------------------------")
# print(f'Training Complete')
# print("------------------------------------------")
# # Creating a directory to save the model paths 

# # Saving the model
# model.save(f'/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/models/dense121_01.h5')
# print("------------------------------------------")
# print(f'Model saved')
# print("------------------------------------------")


# #plotting the accuracy and loss
# print("------------------------------------------")
# print(f'Plotting and supplimentary data')
# print("------------------------------------------")
# plt.figure(figsize=(10, 10))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_acc'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend(['train', 'test'], loc='upper left')
# plt.tight_layout()
# plt.savefig(f'/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/models/Accuracy.jpg')

loaded_model = load_model(f'/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/models/dense121_01.h5')
outcomes = loaded_model.predict(valid_generator)
target_names = ['Cell Space','WhiteSpace']
y_pred = np.argmax(outcomes, axis=1)
# confusion matrix
confusion = confusion_matrix(valid_generator.classes, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/models/Confusion_matrix.jpg')

conf_df = pd.DataFrame(confusion, index = target_names, columns = target_names)
conf_df.to_csv(f'/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/models/Confusion_matrix.csv')

# classification report
report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(f'/home/chs.rintu/Documents/chs-lab-ws02/research-cancerPathology/EffnetDifferentiator/models/Classification_report.csv')


print("------------------------------------------")
print(f'Supplimentary Data Saved')
print("------------------------------------------")