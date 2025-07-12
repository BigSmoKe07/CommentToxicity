# 0. Install Dependencies and Import Libraries
# Run this once in your environment
# !pip install tensorflow tensorflow-gpu pandas matplotlib scikit-learn gradio jinja2

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from matplotlib import pyplot as plt
import gradio as gr

# 1. Load Dataset
# Correct the path here - train.csv inside folder, not repeated twice
DATA_PATH = os.path.join('jigsaw-toxic-comment-classification-challenge', 'train.csv')
df = pd.read_csv(DATA_PATH)

# Show first 5 rows
print(df.head())

# 2. Preprocess Data

# Extract features (comments) and targets (toxic labels)
X = df['comment_text'].values
y = df[df.columns[2:]].values  # columns 2 onwards are label columns

# Set max vocabulary size and sequence length
MAX_FEATURES = 200_000
MAX_LEN = 1800

# Initialize TextVectorization layer for text tokenization and integer encoding
vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=MAX_LEN
)

# Adapt vectorizer to our text data (build vocabulary)
vectorizer.adapt(X)

# Vectorize all comments to integer sequences
vectorized_text = vectorizer(X)

# Create TensorFlow dataset from vectorized text and labels
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))

# Cache, shuffle, batch, and prefetch for performance optimization
dataset = dataset.cache()
dataset = dataset.shuffle(buffer_size=160_000, reshuffle_each_iteration=True)
dataset = dataset.batch(16)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Calculate dataset size for splitting
dataset_size = len(df)

# Create training, validation, and test splits (70%, 20%, 10%)
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.2)
test_size = dataset_size - train_size - val_size

train = dataset.take(train_size // 16)  # divided by batch size
val = dataset.skip(train_size // 16).take(val_size // 16)
test = dataset.skip((train_size + val_size) // 16).take(test_size // 16)

# 3. Build the Model

model = Sequential([
    Embedding(MAX_FEATURES + 1, 32),                    # Embedding layer converts word IDs to vectors
    Bidirectional(LSTM(32, activation='tanh')),         # Bidirectional LSTM for sequence learning
    Dense(128, activation='relu'),                       # Fully connected layers for feature extraction
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')                       # Output layer for 6 labels with sigmoid activation
])

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer='adam'
)

# Print model architecture summary
model.summary()

# 4. Train the Model

history = model.fit(
    train,
    epochs=1,
    validation_data=val
)

# Plot training and validation loss/accuracy curves
plt.figure(figsize=(8, 5))
pd.DataFrame(history.history).plot()
plt.show()

# 5. Make Predictions

# Example comment to test prediction
example_comment = "You freaking suck! I am going to hit you."
input_vector = vectorizer([example_comment])  # Vectorize input comment

# Predict toxicity scores
prediction = model.predict(input_vector)

# Apply threshold to get binary labels
binary_prediction = (prediction > 0.5).astype(int)
print(f"Predicted toxicity labels: {binary_prediction}")

# 6. Evaluate Model Performance on Test Set

# Initialize metrics
precision = Precision()
recall = Recall()
accuracy = CategoricalAccuracy()

# Iterate through test batches and update metrics
for batch_x, batch_y in test.as_numpy_iterator():
    y_pred = model.predict(batch_x)
    
    # Flatten to 1D arrays for metric updates
    y_true_flat = batch_y.flatten()
    y_pred_flat = y_pred.flatten()
    
    precision.update_state(y_true_flat, y_pred_flat)
    recall.update_state(y_true_flat, y_pred_flat)
    accuracy.update_state(y_true_flat, y_pred_flat)

# Print final evaluation results
print(f"Precision: {precision.result().numpy():.4f}")
print(f"Recall: {recall.result().numpy():.4f}")
print(f"Accuracy: {accuracy.result().numpy():.4f}")

# 7. Save and Load Model

# Save model to file
model.save('toxicity.h5')

# Load model from file
loaded_model = tf.keras.models.load_model('toxicity.h5')

# 8. Create Gradio Interface for Live Testing

def score_comment(comment):
    """
    Takes a comment string, vectorizes it, predicts toxicity,
    and returns formatted results for each label.
    """
    vec_comment = vectorizer([comment])
    preds = loaded_model.predict(vec_comment)[0]
    
    result_text = ""
    for idx, col in enumerate(df.columns[2:]):
        label_pred = preds[idx] > 0.5
        result_text += f"{col}: {label_pred}\n"
    return result_text

# Create Gradio interface with textbox input and text output
interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(lines=2, placeholder="Enter comment to score"),
    outputs="text"
)

# Launch the interface with sharing enabled
interface.launch(share=True)
