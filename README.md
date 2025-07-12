# Toxic Comment Classification with TensorFlow

This project builds a multi-label text classification model to detect toxic comments using the Jigsaw Toxic Comment Classification dataset. The model uses a Bidirectional LSTM neural network to classify comments into six toxicity categories.

---

## Features

* Data preprocessing with TensorFlow's `TextVectorization`
* Multi-label classification with a Bidirectional LSTM model
* Training, validation, and testing dataset splits
* Model evaluation using Precision, Recall, and Accuracy
* Save/load model functionality
* Interactive demo using Gradio for live toxicity scoring

---

## Installation

Make sure you have Python 3.9+ installed.

Install dependencies via pip:

```bash
pip install tensorflow tensorflow-gpu pandas matplotlib scikit-learn gradio jinja2
```

---

## Dataset

Download the [Jigsaw Toxic Comment Classification Challenge dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place the `train.csv` file under:

```
jigsaw-toxic-comment-classification-challenge/train.csv
```

---

## Usage

1. **Load and preprocess data**

   * Reads the CSV file.
   * Vectorizes text comments using a vocabulary of 200,000 words.
   * Creates TensorFlow datasets for training, validation, and testing.

2. **Build the model**

   * Embedding layer for word vectors.
   * Bidirectional LSTM layer.
   * Dense layers for feature extraction.
   * Final output layer with sigmoid activation for multi-label output.

3. **Train the model**

   * Trains for one epoch (adjustable).
   * Validates on the validation split.

4. **Evaluate the model**

   * Calculates Precision, Recall, and Accuracy on test data.

5. **Save and load model**

6. **Run inference**

   * Predict toxicity labels for new comments.

7. **Launch Gradio interface**

   * Provides a web UI to input comments and view toxicity predictions interactively.

---

## Example Prediction

```python
comment = "You freaking suck! I am going to hit you."
vectorized_comment = vectorizer([comment])
prediction = model.predict(vectorized_comment)
print((prediction > 0.5).astype(int))
```

---

## Running Gradio Demo

Run the last cell to launch a Gradio web interface. You can enter comments and get live toxicity scores for six categories.

---

## Notes

* Adjust the number of epochs for better training.
* Dataset path must be correctly set for data loading.
* Model saves to `toxicity.h5` in the working directory.

---
