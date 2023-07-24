# House Price Prediction using PCA and Linear Regression

This repository contains code for a house price prediction project using Principal Component Analysis (PCA) for dimensionality reduction and Linear Regression for modeling. The goal is to predict the sale prices of houses based on various features.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Data](#data)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we use PCA to reduce the dimensionality of the feature space and then train a Linear Regression model to predict house prices. PCA helps in capturing the most important patterns in the data while reducing the number of features. The model is trained on the training data and then used to predict the sale prices of houses in the test data.

## Dependencies

To run the code in this repository, you will need the following dependencies:

- Python 3.x
- NumPy
- pandas
- scikit-learn

You can install the required packages using `pip`:

## bash
pip install numpy pandas scikit-learn

## Usage
Clone the repository:

git clone https://github.com/your_username/house-price-prediction.git
cd house-price-prediction

Download the data files (train.csv and test.csv) and place them in the same directory as the code.

Run the main script: python main.py
## Data
The dataset used in this project contains various features of houses, such as zoning information, lot shape, neighborhood, etc., along with their corresponding sale prices. The data is divided into two files:

train.csv: Training data containing the features and sale prices for known houses.
test.csv: Test data containing the features of houses for which we need to predict the sale prices.
Modeling
Data Preprocessing: Missing values in the dataset are filled with mean values, and categorical columns are one-hot encoded.

PCA: Principal Component Analysis is applied to reduce the dimensionality of the feature space.

Model Training: Linear Regression is used as the prediction model.

## Results
After running the code, the predicted sale prices for the houses in the test set will be saved in a CSV file named predictions.csv in the same directory.

## Contributing
Contributions to this project are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. 
