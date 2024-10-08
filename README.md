# House Prices Competition (Kaggle)

This repository contains the solution for the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The goal is to predict the sales prices of houses using a variety of regression techniques based on provided features.

## Table of Contents

- [Project Overview](#project-overview)
- [Files in the Repository](#files-in-the-repository)
- [Setup](#setup)
- [Pipeline](#pipeline)

## Project Overview

The competition requires building models that accurately predict the sales price of a house using features such as area, the number of rooms, year built, and more. This solution leverages feature engineering, exploratory data analysis (EDA), and various machine learning models to find the best predictive performance.

## Files in the Repository

### Data
- `data_description.txt`: Provides details about the dataset used in this project.
- `sample_submission.csv`: Sample of the expected submission format for the Kaggle competition.
- `train.csv`: Training data containing features and target variables (house prices).
- `test.csv`: Test data without target values, used for final predictions.
- `clean_data/`: Directory containing cleaned versions of the datasets used in the project.

### house_prices_kaggle
- `main.py`: Main script to run the prediction pipeline.
- `utils.py`: Utility functions used throughout the project.
- `model/`: Contains saved model and prediction files.
  - `house_prices_regressor.pkl`: Trained model saved as a pickle file.
  - `house_prices_prediction.csv`: Final predictions for the Kaggle competition.

### Notebooks
- `data_preprocessing.ipynb`: Notebook for handling missing data, outliers, and data cleaning.
- `eda.ipynb`: Notebook for exploratory data analysis (EDA), visualizations, and insights on the dataset.
- `feature_engineering.ipynb`: Notebook for creating new features and feature transformation to improve model performance.
- `modeling.ipynb`: Notebook showcasing different models and evaluating their performance.

### Configuration
- `pyproject.toml`: Configuration file for managing project dependencies and settings using Poetry.
- `poetry.lock`: Lock file generated by Poetry to ensure consistent dependencies across environments.
- `.gitignore`: Specifies which files and directories should be ignored by Git version control.

### Documentation
- `README.md`: This file, containing an overview of the project and setup instructions.

## Setup

To set up the environment and install the dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-prices-kaggle.git
   cd house-prices-kaggle

2. Install Poetry, if you don't have it installed already.
    ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   
3. Install Dependencies
    ```bash
   poetry install
   
4. Activate the Virtual Environment
    ```bash
   poetry shell

## Pipeline

The model training and prediction process is structured as follows:

1. **Data Preprocessing and Model Training:**
   - The full pipeline, including data preprocessing, feature engineering, and model training, is implemented in the `pipeline.py` file. This script handles the entire workflow and saves the trained model (`house_prices_regressor.pkl`) for future use.

2. **Prediction:**
   - The trained model is then used in the `main.py` file to generate predictions for the test dataset. These predictions are saved as `house_prices_prediction.csv`, which can be directly submitted to the Kaggle competition.



   
