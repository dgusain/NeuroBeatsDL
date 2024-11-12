import os
import numpy as np
from PIL import Image as IM
import glob
import matplotlib.pyplot as plt
import os
import h5py  # Import HDF5 support
import keras
from keras import layers
import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import tensorflow as tf
from tensorflow.keras.utils import Sequence

# DATA PREPARATION
# fpath = '/projects/academic/courses/cse676s24/dsingh27/CVIP/CNNLSTM/dataset/dataset/preprocessed_data/preprocessed_dataResized100Frames.npy'
fpath = '/projects/academic/courses/cse676s24/dsingh27/CVIP/CNNLSTM/dataset/dataset/preprocessed_data/1003_500sec_8epoch.hdf5'

with h5py.File(fpath, 'r') as h5f:
    dataset = h5f['data'][:]

print(dataset.shape) 

indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[:int(0.8 * len(dataset))]
val_index = indexes[int(0.8 * len(dataset)): int(0.9 * len(dataset))]
test_index = indexes[int(0.9 * len(dataset)):]

def create_shifted_frames(data):
    x = data[:, :-1, :, :, :]
    y = data[:, 1:, :, :, :]
    return x, y

class DataGenerator(Sequence):
    def __init__(self, file_path, dataset_name, indexes, batch_size=1):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.indexes = indexes
        self.batch_size = batch_size

    def __len__(self):
        # Return the number of batches per epoch
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indexes))
        batch_indexes = self.indexes[start:end]
        
        # Initialize arrays for X (features) and y (labels)
        x, y = [], []
        with h5py.File(self.file_path, 'r') as f:
            # Extract a batch of data using the specified indexes
            batch = f[self.dataset_name][batch_indexes]
            x, y = create_shifted_frames(batch)
        
        return np.array(x), np.array(y)

    def on_epoch_end(self):
        # Updates indexes after each epoch
        np.random.shuffle(self.indexes)

file_path2 = '/projects/academic/courses/cse676s24/dsingh27/CVIP/CNNLSTM/dataset/dataset/preprocessed_data/cleanedData500Best.hdf5'
data_name = 'cleaned_data'

# Save the dataset and indices to an HDF5 file
with h5py.File(file_path2, 'w') as f:
    # Create a dataset in the file for the main dataset
    f.create_dataset(data_name, data=dataset, compression='gzip')
    # Create datasets in the file for the indices
    f.create_dataset('train_index', data=train_index, compression='gzip')
    f.create_dataset('val_index', data=val_index, compression='gzip')
    f.create_dataset('test_index', data=test_index, compression='gzip')
    
    
# Load the indices from the HDF5 file
with h5py.File(file_path2, 'r') as f:
    train_index = f['train_index'][:]
    val_index = f['val_index'][:]
    test_index = f['test_index'][:]

# Instantiate the DataGenerators with the data name and indexes
train_generator = DataGenerator(file_path2, data_name, train_index, batch_size=1)
val_generator = DataGenerator(file_path2, data_name, val_index, batch_size=1)
test_generator = DataGenerator(file_path2, data_name, test_index, batch_size=1)

sample_batch_x, _ = train_generator[0]

# MODEL DEFINITION
import keras
from keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(train_generator, val_generator, sample_batch_x):

    inp = layers.Input(shape=(None, *sample_batch_x.shape[2:]))


    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=3,
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same"
    )(x)

    # Setup the model
    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='mse')

    # Callbacks for saving the model and early stopping
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', patience=3)

    # Train the model
    model.fit(
        train_generator,
        epochs=40,
        validation_data=val_generator,
        callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]
    )
    return model

# Usage of the function to train the model
model = train_model(train_generator, val_generator, sample_batch_x)
path_to_save_weights = 'CNNLSTM100Samplesv4_8Frames.h5'

# Save the model weights
model.save_weights(path_to_save_weights)

# Loading the model
import keras
from keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def create_model(sample_batch_x):
    inp = layers.Input(shape=(None, *sample_batch_x.shape[2:]))

    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=3,
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same"
    )(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='mse')

    return model

# Load the model with the same architecture
# sample_batch_x = None  # Make sure to set this with the correct shape that matches training
model = create_model(sample_batch_x)

# Path to your saved weights
path_to_saved_weights = 'CNNLSTM100Samplesv4_8Frames.h5'

# Load the weights
model.load_weights(path_to_saved_weights)

# Now your model is ready to make predictions

# Direct brain map for binaural prediction for 1 second from model weights
test_x, test_y = next(iter(test_generator))
predictions = model.predict(test_x)

# regeneration of brain map quality
x = dataset[:, :-1, :, :, :]
y =  dataset[:, 1:, :, :, :]
reshaped_x = x[0].reshape((1, 7, 256, 256, 3))
reshaped_y = y[0].reshape((1, 7, 256, 256, 3))
pred = reshaped_x

for i in range(2):
    pred = model.predict(pred)
for i in range(7):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8)) 
    
    axs[0].imshow(pred[0][i])
    axs[0].set_title('Predictions')
    
    axs[1].imshow(reshaped_y[0][i])
    axs[1].set_title(f'Actual Data after {i + 1} time seconds')
    
   
    plt.show()

# Regeneration of brain map activity for 7 timeseconds
test_x, test_y = next(iter(test_generator))
x = dataset[:, :-1, :, :, :]
reshaped_x = x[0].reshape((1, 7, 256, 256, 3))
reshaped_y = x[1].reshape((1, 7, 256, 256, 3))
pred = reshaped_x
for i in range(1):
    pred = model.predict(pred)

for i in range(7):
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))  # Increase figure size here
    
    # Plot the first image (prediction) on the left
    axs[0].imshow(pred[0][i])
    axs[0].set_title('Prediction binarual  brain Map')
    
    # Plot the second image (actual data) in the middle
    axs[1].imshow(reshaped_y[0][i])
    axs[1].set_title(f'Actual Binaural Brain Map at {i + 1} time seconds')
    
    # Plot the third image (input data) on the right
    axs[2].imshow(reshaped_x[0][i])
    axs[2].set_title('Input Brain Map wihtout Binarual')
    
    plt.show()


