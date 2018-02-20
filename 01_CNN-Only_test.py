import h5py
import numpy as np
import os
from os.path import join
from keras import layers
from keras.models import Model
from keras.utils import to_categorical

X = {}
Y = {}
with h5py.File(join('datasets', 'UCF11', 'data_frames_50_h_30_w_40.h5'), 'r') as hf:
    X['train'] = hf['X_train'][:]
    X['test'] = hf['X_test'][:]
    X['valid'] = hf['X_valid'][:]
    Y['train'] = hf['Y_train'][:]
    Y['test'] = hf['Y_test'][:]
    Y['valid'] = hf['Y_valid'][:]

h, w, ch = X['train'].shape[2:]
for key in X:
    X[key] = np.reshape(X[key], (-1, h, w, ch)).astype(np.float) / 255
    Y[key] = np.array([[l] * 50 for l in Y[key]])
    Y[key] = to_categorical(np.reshape(Y[key], (-1)))

nb_classes = Y['train'].shape[1]


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
model.fit(X['train'], Y['train'], validation_data=(X['valid'], Y['valid']), epochs=1)

print(model.evaluate(X['test'], Y['test']))




