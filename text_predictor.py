import re
import string

import numpy as np
import tensorflow as tf
from keras.layers import Dense, TextVectorization, Embedding, Dropout, GlobalAveragePooling1D, Activation

# Set random seed for reproducibility
seed = 42
tf.random.set_seed(seed)

# Define the directories for training and testing data
train_dir = 'train'
test_dir = 'test'

# Load the training and testing data
batch_size = 32

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size)

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])


# Create a custom standardization function
# @tf.keras.utils.register_keras_serializable()
# def custom_standardization(input_data):
#     lowercase = tf.strings.lower(input_data)
#     stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
#     return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


# Define the vectorization layer
max_features = 10000
sequence_length = 250


vectorize_layer = TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# Vectorize the text data
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Define the model architecture
embedding_dim = 16
num_classes = 4

model = tf.keras.Sequential([
    Embedding(max_features + 1, embedding_dim),
    Dropout(0.2),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(num_classes)])

model.summary()

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_ds)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    Activation('sigmoid')
])

export_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

export_model.save("text_categorization_model", save_format='tf')



