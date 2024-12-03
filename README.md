# House Price Prediction using Linear Regression with Regularization

## Overview
This repository contains a **step-by-step implementation** of a linear regression model for predicting house prices using **MATLAB**. The model uses various features such as square footage, number of bedrooms, and neighborhood type to predict house prices. L2 regularization (Ridge Regression) is applied to reduce overfitting and improve model generalization.

### Key Features
- **Linear Regression:** Predict house prices based on features like square footage, number of bedrooms, and neighborhood.
- **L2 Regularization (Ridge Regression):** Regularization is applied to prevent overfitting by adding a penalty term to the cost function.
- **Step-by-Step Implementation:** The entire process from data preprocessing to model evaluation is implemented step by step, without using built-in functions for regression.

## Dataset
The model is trained on a dataset containing the following features:
- `Home`: Home identifier (excluded from model features).
- `Price`: Target variable, the house price.
- `SqFt`: Size of the house in square feet.
- `Bedrooms`: Number of bedrooms in the house.
- `Bathrooms`: Number of bathrooms in the house.
- `Offers`: Number of offers received on the house.
- `Brick`: Categorical variable indicating whether the house has a brick exterior (Yes/No).
- `Neighborhood`: Categorical variable indicating the neighborhood the house is located in.

## Steps Implemented
### 1. **Data Preprocessing:**
   - Removal of non-predictive columns (e.g., `Home`).
   - Encoding categorical features like `Brick` and `Neighborhood` using one-hot encoding.
   - Normalization of numerical features (e.g., `SqFt`, `Bedrooms`, `Bathrooms`, `Offers`).

### 2. **Model Training:**
   - Split the dataset into training and test sets.
   - Implemented **Linear Regression** using gradient descent.
   - Applied **L2 Regularization** (Ridge Regression) to the cost function to prevent overfitting.

### 3. **Model Evaluation:**
   - Calculated **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** to evaluate the model’s performance.
   - Visualized the convergence of the cost function during training.
   - Plotted actual vs predicted house prices for evaluation.

## Getting Started
To run the code on your local machine, follow these steps:

### Prerequisites
- MATLAB (preferably the latest version for compatibility)
- Dataset file `house_prices.csv` (this file should be in the same directory as the script)

### Installation
1. Download or clone the repository to your local machine:
   ```bash
   git clone https://github.com/ahmdmohamedd/house-price-prediction-linear-regression.git
   ```
2. Open the `house_price_prediction.m` script in MATLAB.
3. Make sure the `house_prices.csv` dataset is in the same directory as the script.
4. Run the script to train the model and evaluate the performance.

### Script Execution
After running the script, the following outputs will be displayed:
- The **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** on the test set.
- A plot showing the convergence of the cost function during gradient descent.
- A scatter plot comparing the actual and predicted house prices.

## Model Tuning
- The **regularization parameter \( \lambda \)** can be adjusted to control the amount of regularization applied. Higher values of \( \lambda \) will result in more regularization.
- You can experiment with different values of \( \lambda \) (e.g., 0.1, 1, 10) to see how it affects the performance of the model.

## Evaluation Metrics
- **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in the predictions.
- **Root Mean Squared Error (RMSE):** Measures the square root of the average squared differences between predicted and actual values.

## Future Improvements
- **Feature Engineering:** Additional features can be added, such as interaction terms between features (e.g., `SqFt per Bedroom`).
- **Polynomial Regression:** Consider using polynomial features to capture non-linear relationships between features and house prices.
- **Hyperparameter Optimization:** Use techniques like cross-validation to tune the learning rate and regularization parameter for optimal performance.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.
