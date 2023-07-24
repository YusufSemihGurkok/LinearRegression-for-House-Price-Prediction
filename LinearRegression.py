import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import math

# Load the data
train = pd.read_csv("train.csv")
test_x = pd.read_csv("test.csv")

# Separate the target variable (SalePrice) from the features in the training set
train_x = train.drop(columns=["SalePrice"])
train_y = train["SalePrice"]

# Fill missing values with mean in both train and test sets
train_x.fillna(train_x.mean(), inplace=True)
test_x.fillna(test_x.mean(), inplace=True)

def one_hot_encode(df, column):
    unique_values = df[column].unique()
    for value in unique_values:
        df[column + '_' + str(value)] = (df[column] == value).astype(int)
    df.drop(column, axis=1, inplace=True)
    return df

# One-hot encode categorical columns in both train and test sets
categorical_columns = ['MSZoning', "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",  "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

for column in categorical_columns:
    train_x = one_hot_encode(train_x, column)
    test_x = one_hot_encode(test_x, column)

# Convert DataFrame to numpy array and float type for PCA
train_x = train_x.values.astype(float)
test_x = test_x.values.astype(float)



def pca(X, k):
    # Calculate the covariance matrix
    cov = np.cov(X, rowvar=False)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort the eigenvalues and corresponding eigenvectors from largest to smallest eigenvalue
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]

    # Project the data onto the top k eigenvectors
    X_transformed = X @ top_k_eigenvectors

    return X_transformed

# Apply PCA to train and test sets
train_x_pca = pca(train_x, k=30)
test_x_pca = pca(test_x, k=30)

# Train the model
betaRSS = np.linalg.inv(train_x_pca.T @ train_x_pca) @ train_x_pca.T @ train_y

# Make predictions on test set
predictions = test_x_pca @ betaRSS

# Round the predictions and take absolute values (if necessary)
predictions = np.round(predictions)
predictions = np.abs(predictions)

# Prepare predictions for submission
id_col = np.arange(1461, 1461 + test_x.shape[0])
predictions_arr = np.column_stack((id_col, predictions))
predictions_arr = predictions_arr.astype(int)

# Save the predictions to a CSV file
predictions_df = pd.DataFrame(predictions_arr, columns=['Id', 'SalePrice'])
print("predictions",predictions_df)
predictions_df.to_csv("predictions9.csv", index=False)

