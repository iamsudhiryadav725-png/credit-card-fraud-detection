Machine Learning-based Credit Card Fraud Detection using Neural Networks (Keras)

A machine learning project for detecting fraudulent credit card transactions using Artificial Neural Networks (ANNs) with Keras/TensorFlow.

The main objective of this project is to classify credit card transactions into two categories:

Genuine Transaction — Class 0
Fraudulent Transaction — Class 1
The project focuses on the challenges of binary classification with a highly imbalanced dataset, where fraudulent transactions represent only a very small percentage of all transactions.

Project Overview
Credit card fraud detection is a binary classification problem in which a machine learning model learns patterns from historical transaction data and predicts whether a new transaction is genuine or fraudulent.

This project uses Neural Networks, specifically Multi-Layer Perceptrons (MLPs), to learn the relationship between transaction features and the fraud label.

The project experiments with different neural network architectures to investigate whether increasing the number of hidden layers affects fraud-detection performance.

Two main models are considered:

MLP with one hidden layer
MLP with two hidden layers
The performance of these models can then be compared using appropriate classification metrics.

Dataset
The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle.

Dataset source:

Kaggle — Credit Card Fraud Detection

The dataset contains 284,807 credit card transactions, collected over a period of two days.

Only 492 transactions are fraudulent, while the remaining transactions are genuine.

This creates a highly imbalanced classification problem.

Dataset Statistics
Property	Value
Total Transactions	284,807
Fraudulent Transactions	492
Genuine Transactions	284,315
Fraud Percentage	~0.172%
Number of Features	31
Target Variable	Class
Dataset Features
The dataset contains 31 columns.

PCA Features
Most of the transaction features are represented by:

V1, V2, V3, ..., V28
These 28 features were transformed using Principal Component Analysis (PCA) to protect sensitive information.

The original transaction attributes were not provided because of confidentiality concerns.

Time
The Time feature represents the number of seconds elapsed between a transaction and the first transaction in the dataset.

Amount
The Amount feature represents the transaction amount.

Class
The Class column is the target variable.

0 → Genuine Transaction
1 → Fraudulent Transaction
Why Is This Dataset Difficult?
The biggest challenge is class imbalance.

Out of 284,807 transactions, only 492 are fraudulent.

That means a model could predict almost every transaction as genuine and still achieve a very high accuracy.

For example, if a model predicted:

Every transaction → Genuine
it would appear highly accurate because fraudulent transactions are extremely rare.

However, such a model would be useless for actual fraud detection because it would fail to identify fraudulent transactions.

Therefore, this project should not rely on accuracy alone.

Machine Learning Approach
The project uses an Artificial Neural Network (ANN) for binary classification.

The overall workflow is:

Credit Card Dataset
        ↓
Data Preprocessing
        ↓
Feature Selection / Preparation
        ↓
Handling Class Imbalance
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Neural Network
        ↓
Model Training
        ↓
Prediction
        ↓
Performance Evaluation
Neural Network Architecture
A neural network consists of several types of layers.

Input Layer
     ↓
Hidden Layer(s)
     ↓
Output Layer
Each neuron receives input values, applies weights and a bias, and passes the result through an activation function.

A simplified neuron can be represented as:

Output = Activation(Weights × Inputs + Bias)
During training, the network adjusts its weights to reduce the difference between predicted and actual results.

Model 1 — One Hidden Layer
The first experiment uses a Multi-Layer Perceptron containing:

Input Layer
     ↓
Hidden Layer
     ↓
Output Layer
The input layer receives the transaction features.

The hidden layer learns patterns within the transaction data.

The output layer produces the probability of the transaction belonging to the fraud class.

For binary classification, the output can be interpreted as:

Probability close to 0 → Genuine
Probability close to 1 → Fraud
Model 2 — Two Hidden Layers
The second experiment increases the model complexity:

Input Layer
     ↓
Hidden Layer 1
     ↓
Hidden Layer 2
     ↓
Output Layer
The purpose of this experiment is to determine whether adding another hidden layer improves the model's ability to identify fraudulent transactions.

The additional hidden layer allows the network to learn more complex representations of the input features.

However, a deeper network does not automatically guarantee better performance. Its effectiveness depends on the architecture, training process, data preprocessing, and evaluation strategy.

Handling the Imbalanced Dataset
Class imbalance is one of the most important aspects of this project.

Because fraud transactions are extremely rare, the model needs to pay particular attention to the minority class.

One approach explored in fraud-classification research is undersampling, where some genuine transactions are removed to create a more balanced training dataset.

The purpose is to prevent the model from being dominated by the majority class.

A simplified representation is:

Original Dataset

Genuine  ███████████████████████████████████████
Fraud    █

             ↓

Balanced / Undersampled Training Data

Genuine  ███████████
Fraud    ███████████
It is important to apply sampling techniques carefully and avoid modifying the test set in a way that gives misleading evaluation results.

Data Preprocessing
Before training the neural network, the transaction data needs to be prepared.

Typical preprocessing steps include:

1. Load the Dataset
The CSV dataset is loaded into a Python data-processing environment.

2. Separate Features and Target
The transaction features are separated from the Class column.

X → Input Features
y → Target Class
3. Feature Scaling
Features such as Amount can have a substantially different numerical scale from other variables.

Scaling helps neural networks train more effectively.

Common approaches include:

Standardization
Normalization
4. Train-Test Split
The dataset is divided into training and testing subsets.

Training Data → Used to learn patterns
Testing Data  → Used to evaluate the trained model
Model Training
The neural network learns from the training data through multiple iterations.

During each training step:

Input Transaction
       ↓
Forward Propagation
       ↓
Prediction
       ↓
Calculate Loss
       ↓
Backpropagation
       ↓
Update Weights
Forward Propagation
The transaction features pass through the neural network from the input layer to the output layer.

Loss Calculation
The predicted result is compared with the actual class.

For binary classification, binary cross-entropy is commonly used as the loss function.

Backpropagation
The model calculates how much each parameter contributed to the prediction error.

The weights are then updated to reduce the loss.

This process repeats over multiple training epochs.

Fraud Prediction
After training, the model can receive a transaction's features and generate a probability.

For example:

Model Output:

0.02 → Likely Genuine
0.91 → Likely Fraud
A classification threshold can then be used to convert the probability into a class prediction.

For example:

Probability < threshold → Class 0
Probability ≥ threshold → Class 1
The appropriate threshold should be selected based on the application's fraud-detection requirements rather than automatically assuming that one threshold is always optimal.

Model Evaluation
Because the dataset is highly imbalanced, multiple evaluation metrics are important.

Confusion Matrix
The confusion matrix contains four possible outcomes:

                    Predicted
                 Genuine   Fraud

Actual Genuine     TN        FP
Actual Fraud       FN        TP
Where:

TP — True Positive: Fraud correctly detected
TN — True Negative: Genuine transaction correctly identified
FP — False Positive: Genuine transaction incorrectly flagged as fraud
FN — False Negative: Fraud transaction incorrectly classified as genuine
Precision
Precision answers:

Of all transactions predicted as fraud, how many were actually fraudulent?

Precision = TP / (TP + FP)
High precision means the system does not flag too many genuine transactions as fraud.

Recall
Recall answers:

Of all actual fraudulent transactions, how many did the model detect?

Recall = TP / (TP + FN)
Recall is particularly important in fraud detection because missing fraudulent transactions can be costly.

F1 Score
The F1 score combines precision and recall.

F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
It provides a balance between precision and recall.

Accuracy
Accuracy measures the overall percentage of correctly classified transactions.

Accuracy = (TP + TN) / Total Transactions
However, accuracy alone can be misleading for this dataset because fraudulent transactions are extremely rare.

Therefore, precision, recall, F1-score, confusion matrix, and potentially PR-AUC/ROC-AUC provide additional insight into model performance.

Experimental Design
The main research question of the project is:

Does changing the number of hidden layers in a neural network significantly affect its performance on credit card fraud detection?

Two neural-network configurations are compared:

Model 1
Input
 ↓
1 Hidden Layer
 ↓
Output
and

Model 2
Input
 ↓
Hidden Layer 1
 ↓
Hidden Layer 2
 ↓
Output
The models are trained and evaluated under comparable conditions.

Their performance can then be compared using classification metrics.

Hypothesis
Null Hypothesis (H0)
There is insufficient evidence to conclude that changing the number of hidden layers significantly affects the performance of the neural network.

Alternative Hypothesis (H1)
Changing the number of hidden layers significantly affects the performance of the neural network.

The experiment investigates whether the observed difference between the two architectures provides evidence in favor of the alternative hypothesis.

Technologies Used
Python
TensorFlow
Keras
NumPy
Pandas
Scikit-learn
Matplotlib
Jupyter Notebook
Kaggle Dataset
How to Run the Project
1. Clone the Repository
git clone YOUR_GITHUB_REPOSITORY_URL
cd Credit-Card-Fraud-Detection
2. Create a Virtual Environment
python -m venv venv
Windows
venv\Scripts\activate
macOS / Linux
source venv/bin/activate
3. Install Required Libraries
If the repository contains a requirements.txt file:

pip install -r requirements.txt
Otherwise, install the main dependencies:

pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
4. Download the Dataset
Download the Credit Card Fraud Detection dataset from Kaggle and place the CSV file in the appropriate project/data directory.

The expected dataset file is commonly:

creditcard.csv
5. Start Jupyter Notebook
jupyter notebook
Open the project's notebook and run the cells sequentially.

Project Workflow
                 ┌──────────────────┐
                 │ Credit Card Data │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Data Preparation │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Feature Scaling  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Class Imbalance   │
                 │ Handling          │
                 └────────┬─────────┘
                          ↓
              ┌─────────────────────────┐
              │ Neural Network Training │
              └────────────┬────────────┘
                           ↓
                 ┌──────────────────┐
                 │ Fraud Prediction │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Model Evaluation │
                 └──────────────────┘
Project Objective
The purpose of this project is not simply to achieve a high accuracy score. The primary objective is to investigate how different Neural Network architectures perform when identifying fraudulent transactions in a highly imbalanced dataset.

The project demonstrates practical concepts including:

Binary classification
Artificial Neural Networks
Keras/TensorFlow
Data preprocessing
Feature scaling
Class imbalance
Undersampling
Model training
Hyperparameter tuning
Confusion matrices
Precision and recall
F1-score
Model comparison
Limitations
This project is primarily an experimental machine-learning implementation using a historical dataset.

Some limitations include:

Fraudulent transactions are extremely rare compared with genuine transactions.
PCA-transformed features make direct interpretation of individual transaction attributes difficult.
Historical transaction patterns may not represent current fraud behavior.
Model performance depends strongly on preprocessing and sampling strategies.
A research/academic model should not automatically be treated as a production fraud-detection system.
Future Improvements
Future versions could improve the system by implementing:

Advanced class-imbalance techniques
SMOTE and other resampling approaches
Precision-Recall curve analysis
ROC-AUC and PR-AUC comparison
Hyperparameter optimization
Ensemble learning
Random Forest and XGBoost comparison
Autoencoder-based anomaly detection
Real-time transaction prediction
Model deployment through a REST API
Monitoring for changes in fraud patterns
Reference
The dataset and research context are based on work by:

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi

Calibrating Probability with Undersampling for Unbalanced Classification.

IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.

Author
Sudhir Yadav

B.Tech — Computer Science and Engineering
