import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, utils
from tensorflow.keras import models, layers, activations, initializers, losses, optimizers, metrics

df = pd.read_csv("../winequality-red.csv")
data = df.drop(columns=['quality']).copy()
target = df.quality.copy()

model = models.Sequential() # Build up the "Sequence" of layers (Linear stack of layers)

# Dense-layer (with he-initialization)
model.add(layers.Dense(input_dim=11, units=40, activation=None, kernel_initializer=initializers.he_uniform())) # he-uniform initialization
# model.add(layers.BatchNormalization()) # Use this line as if needed
model.add(layers.Activation('relu')) # elu or relu (or layers.ELU / layers.LeakyReLU)

model.add(layers.Dense(units=20, activation=None, kernel_initializer=initializers.he_uniform())) 
model.add(layers.Activation('relu'))

model.add(layers.Dense(units=10, activation=None, kernel_initializer=initializers.he_uniform())) 
model.add(layers.Activation('relu'))
# model.add(layers.Dropout(rate=0.4)) # Dropout-layer

model.add(layers.Dense(units=1, activation=None)) 

model.compile(optimizer=optimizers.Adam(), # Please try the Adam-optimizer
              loss=losses.mean_squared_error, # MSE 
              metrics=[metrics.mean_squared_error]) # MSE
            
history = model.fit(train_data, train_target, batch_size=100, epochs=1000, validation_split=0.3, verbose=0)
