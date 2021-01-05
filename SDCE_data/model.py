import os
import csv

import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D,MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)    
    # skip the first line (header)
    next(reader)   
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# if I want to double the number of images for traiing I can do this:
# - flip the images horizontally
# - reverse measurement sign

'''
import numpy as np
image_flipped = np.fliplr(image)
measurement_flipped = -measurement
'''

def preProcess(color_images):
    # grayscale
    grayscaled_images = np.sum(color_images/3, axis=3, keepdims=True)
    
    # normalize
    normalized_images = (grayscaled_images - 128) / 128
    
    
    return normalized_images

#NOTE: cv2.imread will get images in BGR format, while drive.py uses RGB. 
#    In the generator below we keep the same image formatting 
#    by doing 
#        "image = ndimage.imread(current_path)" with "from scipy import ndimage"

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # add Center view
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # add flipped:
                flipped_center_image = np.fliplr(center_image)
                images.append(flipped_center_image)
                angles.append(-1.0 * center_angle)
                
                # add Left view
                name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = float(batch_sample[3]) + 0.2
                images.append(left_image)
                angles.append(left_angle)
                # add flipped:
                flipped_left_image = np.fliplr(left_image)
                images.append(flipped_left_image)
                angles.append(-1.0 * left_angle)
                
                # add Right view
                name = '/opt/carnd_p3/data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = float(batch_sample[3]) - 0.2
                images.append(right_image)
                angles.append(right_angle)
                # add flipped:
                flipped_right_image = np.fliplr(right_image)
                images.append(flipped_right_image)
                angles.append(-1.0 * right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            #print(X_train[0].shape)
            yield sklearn.utils.shuffle(X_train, y_train)




# Set our batch size
batch_size=64

'''
# very simple network
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten(input_shape = (160, 320, 3)))
model.add(Dense(1))

# Lenet network:
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#if I want to crop
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
'''

#NVDIA network (add Dropouts at will):
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2,2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2,2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2,2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))


#model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer=Adam(0.0005))


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

callbacks = [
        EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint(filepath='model.h5', save_best_only=True, monitor='val_loss')
]

model.fit_generator(train_generator, steps_per_epoch=int((len(train_samples) * 6) / batch_size),
                                      validation_data=validation_generator,
                                      validation_steps=int((len(validation_samples) * 6) / batch_size),
                                      epochs=10, verbose=1, callbacks=callbacks)
