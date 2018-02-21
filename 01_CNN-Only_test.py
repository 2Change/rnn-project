import h5py
import numpy as np
import os
import random
from os.path import join
from keras import layers
from keras.models import Model
from keras.utils import to_categorical

if False:
    X = {}
    Y = {}
    with h5py.File(join('datasets', 'UCF11', 'data_frames_50_h_120_w_160.h5'), 'r') as hf:
        X['train'] = hf['X_train'][:]
        X['test'] = hf['X_test'][:]
        X['valid'] = hf['X_valid'][:]
        Y['train'] = hf['Y_train'][:]
        Y['test'] = hf['Y_test'][:]
        Y['valid'] = hf['Y_valid'][:]

    print('Loaded data from h5py dataset')

    h, w, ch = X['train'].shape[2:]
    for key in X:
        X[key] = np.reshape(X[key], (-1, h, w, ch)).astype(np.float) / 255
        Y[key] = np.array([[l] * 50 for l in Y[key]])
        Y[key] = to_categorical(np.reshape(Y[key], (-1)))

    nb_classes = Y['train'].shape[1]



h, w, ch = 120, 160, 3
nb_classes = 11

def train_generator(dataset_dir, batch_size):
    
    all_files = os.listdir(dataset_dir)
    
    while True:
        images = []
        labels = []

        for _ in range(batch_size):
            
            random_filename = random.choice(all_files)
            
            with h5py.File(join(dataset_dir, random_filename), 'r') as hf:
                frames = hf['X'][:]
                fr_labels = hf['Y'][:]
                random_idx = np.random.randint(frames.shape[0])
                
                images.append(frames[random_idx])
                labels.append(fr_labels)
                
        images = np.array(images)
        labels = to_categorical(np.array(labels), nb_classes)
        
        yield images, labels
        
        

def valid_generator(dataset_dir, batch_size):
    
    all_files = os.listdir(dataset_dir)
        
    for filename in all_files:

        with h5py.File(filename, 'r') as hf:
            frames = hf['X'][:]
            single_label = hf['Y'][:][0]
            
            fr_labels = np.array([single_label] * frames.shape[0])

            yield frames, to_categorical(fr_labels, nb_classes)


        
separate_dataset_dir = join('datasets', 'UCF11', 'separate_frames_50_h_120_w_160')
batch_size = 32

# build model
input_layer = layers.Input((h, w, ch))
x = input_layer
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(nb_classes, activation='softmax')(x)
model = Model(inputs=[input_layer], outputs=[x])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(X['train'], Y['train'], validation_data=(X['valid'], Y['valid']), epochs=1)

model.fit_generator(train_generator(join(separate_dataset_dir, 'train'), batch_size), steps_per_epoch=947 * 50 // batch_size, 
                    validation_data=valid_generator(join(separate_dataset_dir, 'valid'), batch_size),
                    validation_steps=318,
                    epochs=5)

#print(model.evaluate(X['test'], Y['test']))