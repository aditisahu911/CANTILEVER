
# Sentiment Analysis System (IMDb Movie Reviews)

This is a small project where I built a sentiment analysis model using LSTM to figure out whether a movie review is *positive* or *negative*. It uses deep learning (LSTM) on the IMDb dataset, which already has the reviews labeled as 1 (positive) or 0 (negative). The goal was to preprocess the text data, build a good model, and improve it using regularization and early stopping.

## Context

Sentiment analysis is about figuring out emotions in text. Here, I tried to build a model that reads a movie review and predicts whether it’s a happy/positive review or an unhappy/negative one.

This project helped me understand how to:
- Use **pre-trained datasets** (like IMDb)
- Do **text preprocessing** with padding
- Build a **Sequential model with Embedding + LSTM**
- Deal with **overfitting** using Dropout and L2 regularization
- Use **EarlyStopping** to avoid training forever
- Evaluate using accuracy, confusion matrix, and classification report


## Dataset

- Dataset used: `IMDb` movie review dataset (comes built-in with Keras)
- Number of words used: Top 10,000 most frequent words
- Review length capped to: 200 tokens
- Total data: 50,000 reviews
  - 25,000 for training
  - 25,000 for testing


## Model Architecture

Here's the basic architecture of the model:

```python
Embedding → LSTM → Dropout → Dense(sigmoid)
```

And the parameters:
- `Embedding` layer: Converts word index into dense vector (dim = 128)
- `LSTM(64)` with L2 regularization
- `Dropout(0.6)` to avoid overfitting
- `Dense(1, activation='sigmoid')` for binary classification

Early stopping was used with patience of 3 to stop training if the validation loss wasn't improving.

## Evaluation

After training, the model was tested on the test set. Some evaluation metrics:

```text
Accuracy Score: 84.66%

Confusion Matrix:
 [[10265  2235]
  [ 1600 10900]]

Classification Report:
               precision    recall  f1-score   support

           0       0.87      0.82      0.84     12500
           1       0.83      0.87      0.85     12500
```

So basically, the model performed quite decently with balanced precision and recall for both classes.


## Training Graphs

The graphs for accuracy and loss over epochs are plotted using matplotlib:

- Accuracy curve shows how the model improved on training and validation data.
- Loss curve helps us see if there was any overfitting.




