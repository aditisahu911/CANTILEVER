
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

---

# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques — primarily Logistic Regression and Random Forest — on an imbalanced dataset. The goal is to accurately identify fraud cases without being misled by the class imbalance, which is common in real-world fraud detection scenarios.

## Context

Fraudulent credit card transactions are a serious concern for financial institutions and customers. Early and accurate detection is essential to minimize financial losses and maintain trust.  
In this project, I’ve implemented anomaly detection and classification models to predict whether a transaction is fraudulent or not, keeping in mind the imbalance in the dataset.

## Content

- **Dataset Source**: Real-world credit card transactions made by European cardholders in September 2013.
- **Total Transactions**: 284,807  
- **Fraudulent Transactions**: 492 (which is only 0.172%)

All features except `Time` and `Amount` have been transformed using PCA for confidentiality. The target variable is `Class`, where:
- `1` = Fraud
- `0` = Legitimate

Due to the extreme imbalance, evaluation metrics like **Precision**, **Recall**, **F1-score**, and **AUC** are more informative than simple accuracy.

## Models Used

- Logistic Regression  
- Random Forest (default settings)  
- Random Forest with hyperparameter tuning (`max_depth=12`)

Each model was evaluated using:
- Confusion Matrix
- Classification Report
- ROC Curve and AUC Score

## Model Performance Summary

| Model                          | Precision | Recall | F1-score | AUC   |
|-------------------------------|-----------|--------|----------|--------|
| Logistic Regression           | 0.06      | 0.88   | 0.12     | 0.9279 |
| Random Forest (default)       | 0.87      | 0.79   | 0.83     | 0.8951 |
| Random Forest (max_depth=12)  | 0.59      | 0.81   | 0.68     | 0.9718 |


## Observations

- **Logistic Regression** has high recall, which means it's good at identifying frauds, but its precision is low — many false positives.
- **Random Forest** (even without tuning) balances precision and recall well.
- **Tuned Random Forest** performs the best overall — especially in terms of AUC and F1-score — making it suitable for this use case.


