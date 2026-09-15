# Credit Card Fraud Detection

A Machine Learning project for detecting fraudulent credit card transactions using **Artificial Neural Networks (ANN)** with **Keras**. The project focuses on binary classification of credit card transactions into **Fraudulent** and **Genuine** categories.

## Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions are extremely rare compared to genuine transactions. This project uses a highly imbalanced credit card transaction dataset and investigates how the architecture of a Neural Network affects fraud detection performance.

The project compares **Multi-Layer Perceptron (MLP)** neural networks with different numbers of hidden layers to determine whether increasing the number of hidden layers significantly changes the model's performance.

## Dataset

The dataset contains **284,807 credit card transactions** made over a period of two days.

* **492 transactions** are fraudulent.
* Fraudulent transactions represent approximately **0.172%** of the total dataset.
* The dataset contains **31 features**.
* **V1–V28** are PCA-transformed features.
* **Time** represents the elapsed time between transactions.
* **Amount** represents the transaction amount.
* **Class** is the target variable:

  * `0` = Genuine transaction
  * `1` = Fraudulent transaction

## Technologies Used

* Python
* Jupyter Notebook
* Keras
* Neural Networks
* Machine Learning
* Data Classification
* Grid Search

## Methodology

The project uses a **Multi-Layer Perceptron (MLP)** based Neural Network for fraud classification.

Two different neural network architectures are investigated:

1. **MLP with one hidden layer**
2. **MLP with two hidden layers**

Model parameters are optimized using **Grid Search**, followed by trial-and-error experimentation. Other parameters are based on the default configurations provided by Keras where applicable.

## Objective

The main objective is to investigate whether changing the number of hidden layers in a Neural Network has a significant effect on the performance of credit card fraud detection.

### Hypothesis

**Null Hypothesis (H0):** There is insufficient evidence to conclude that the number of hidden layers significantly affects model performance.

**Alternative Hypothesis (H1):** Changing the number of hidden layers significantly affects the performance of the Neural Network.

## Project Structure

* `credit-card-fraud-detection-using-neural-networks.ipynb` — Jupyter Notebook containing the data analysis, model implementation and experiments.
* `README.md` — Project documentation.

## Dataset Source

The dataset was obtained from the Credit Card Fraud Detection dataset available on Kaggle.

## Key Learning Outcomes

* Handling highly imbalanced classification datasets
* Understanding fraud detection using Machine Learning
* Implementing Artificial Neural Networks with Keras
* Comparing different Neural Network architectures
* Using Grid Search for parameter optimization
* Working with PCA-transformed features
* Understanding binary classification problems

## References

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.

