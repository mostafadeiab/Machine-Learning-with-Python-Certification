# -*- coding: utf-8 -*-
# import libraries
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# get data files
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary (ham=0, spam=1)
train_data['label'] = train_data['label'].apply(lambda x: 1 if x == 'spam' else 0)
test_data['label'] = test_data['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the data into features and labels
train_messages = train_data['message']
train_labels = train_data['label'].values
test_messages = test_data['message']
test_labels = test_data['label'].values

# Tokenize the text data
tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_messages)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_messages)
test_sequences = tokenizer.texts_to_sequences(test_messages)

# Pad the sequences
max_length = 100
train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

model = keras.Sequential([
    keras.layers.Embedding(10000, 16, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
  sequence = tokenizer.texts_to_sequences([pred_text])
  padded_sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
  
  # Predict the class
  prediction = model.predict(padded_sequence)[0][0]
  
  # Determine the label
  label = "spam" if prediction > 0.5 else "ham"
  
  return [prediction, label]

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()