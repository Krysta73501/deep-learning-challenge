# deep-learning-challenge

# Introduction
This project uses machine learning to predict charitable donations based on provided data. The model is built using TensorFlow and includes steps for data preprocessing, model training, and evaluation.

# Installation
To install the necessary dependencies, run:

pip install pandas scikit-learn tensorflow

# Usage
Import the necessary libraries:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
Load the dataset:

application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
application_df.head()
Preprocess the data by dropping non-beneficial columns:

application_df.drop(columns=['EIN', 'NAME'], inplace=True)

# Features
Data Preprocessing: Cleaning and preparing the data for model training.
Model Training: Building and training a neural network using TensorFlow.
Model Evaluation: Assessing the performance of the trained model.

# Dependencies
pandas
scikit-learn
tensorflow

# Contributors
Krysta Sharp

# Acknowledgment
This model was completed withg thge assistance of ChatGPT, a language model developed by OpenAI to help correct any coding errors that couldn't be resolved.
