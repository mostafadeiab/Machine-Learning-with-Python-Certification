# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# Import libraries. You may or may not use all of these.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import data
dataset = pd.read_csv('insurance.csv')
dataset.tail()

dataset = pd.get_dummies(dataset)
x = dataset.drop(columns=['expenses'])
y = dataset['expenses']

# Split the data into training (80%) and testing (20%) sets
train_dataset, test_dataset, train_labels, test_labels = train_test_split(x, y, test_size=0.2)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[train_dataset.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(
    optimizer='Adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae', 'mse']  # Mean Absolute Error
)

history = model.fit(
    train_dataset , train_labels,
    epochs=500,
    callbacks=[tfdocs.modeling.EpochDots()],
    verbose=0
)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)