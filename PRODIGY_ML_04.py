#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from random import shuffle
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
import random
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import RMSprop, Adam
from tensorflow.keras import layers, models
from keras.models import save_model
from keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D
from tensorflow.keras.utils import to_categorical


# In[ ]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[ ]:


get_ipython().system('kaggle datasets download -d gti-upm/leapgestrecog')


# In[ ]:


import zipfile

zip_ref= zipfile.ZipFile('/content/leapgestrecog.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()


# In[ ]:


path= '/content/leapGestRecog'
hand_gestures = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]
final_data = [] # An empty list to hold the dictionaries for each gesture.
image_size = (150, 150)

for file in range(10): # Looping through the 10 files containing the gestures.
    file_data = os.path.join(path, f"{file:02d}")
    for gesture in hand_gestures: # Looping through the gestures files.
        gestures_img = os.path.join(file_data, gesture)
        if os.path.exists(gestures_img):
            for filename in os.listdir(gestures_img): # Looping through the images in each gestures file.
                if filename.endswith((".png", ".jpg")):
                    image_path = os.path.join(gestures_img, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None: # Checking if it's a valid image
                        resized_image = cv2.resize(image, image_size)
                        final_data.append({"gesture": gesture, "image": resized_image})


# In[ ]:


total_count = len(final_data)
print(f"total data count: {total_count}")


# In[ ]:


images = np.array([item['image'] for item in final_data])
labels = np.array([item['gesture'] for item in final_data])

images = np.array(images, dtype = 'float32')
labels = np.array(labels)
labels = labels.reshape(total_count, 1)


# In[ ]:


fig,ax = plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l = random.randint(0,len(labels))
        ax[i,j].imshow(images[l])
plt.tight_layout()


# In[ ]:


# Conversion of labels to numbers

label_encoder= LabelEncoder()
labels= label_encoder.fit_transform(labels)
labels= to_categorical(labels)

# Splitting train and test data

X_train, X_test, y_train, y_test= train_test_split(images, labels, test_size= 0.2, random_state= 42)


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


# CNN

model= models.Sequential()

# 1st layer
model.add(Conv2D(filters= 15, kernel_size= (5,5), padding= 'Same',
                 activation= 'relu', input_shape= (150, 150, 1)))
model.add(MaxPool2D(pool_size= (2,2)))
model.add(Dropout(0.25))

# 2nd layer
model.add(Conv2D(filters= 15, kernel_size= (5,5), padding= 'Same',
                 activation= 'relu'))
model.add(MaxPool2D(pool_size= (2,2), strides= (2,2)))
model.add(Dropout(0.25))

# Flatten
model.add(Flatten())

model.add(Dense(512, activation= 'relu'))
model.add(Dense(len(hand_gestures), activation= 'softmax'))

# Optimizer
optimizer= Adam(learning_rate= 0.001, beta_1= 0.9, beta_2= 0.999)

# Compile
model.compile(optimizer= optimizer, loss= 'categorical_crossentropy', metrics= ['accuracy'])

# Early stopping
early_stopping= EarlyStopping(monitor= 'val_loss', patience= 1, restore_best_weights= True)

# Training
history = model.fit(X_train, y_train, epochs = 3, batch_size = 250, validation_data = (X_test, y_test), callbacks = [early_stopping])

test_loss, test_acc= model.evaluate(X_test, y_test)
print("Accuracy: ", test_acc)


# In[ ]:


# Plotting the accuracy
def plot(history):
    # The training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper right')
    plt.show()

    # The training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc = 'upper right')
    plt.show()

# Calling the function
plot(history)


# In[ ]:




