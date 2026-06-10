# Bank Term Deposit Prediction

A Streamlit web application for predicting whether a bank customer will subscribe to a term deposit based on direct marketing campaign data.

## Overview

This project demonstrates a full machine learning workflow for bank term deposit subscription prediction, including data preprocessing, model training, evaluation, and deployment.

The app uses a trained CatBoost pipeline to predict customer conversion likelihood and provides model performance and visualization summaries.

## Tech Stack

- Python
- Streamlit
- pandas, numpy
- scikit-learn
- CatBoost
- TensorFlow / Keras
- joblib
- Pillow

## Project Structure

- `App/app.py` - Streamlit application entrypoint.
- `Code/` - Jupyter notebooks for data preprocessing, machine learning modeling, deep learning modeling, and experimentation without a pipeline.
- `Data/` - Source dataset files used for training and analysis.
- `Image/` - Visualization assets shown in the Streamlit app.
- `Model/` - Saved model pipelines and feature metadata.
- `BankDepositSubscriptionPrediction.docx` - Project report or documentation.

## Key Features

- Interactive user input form for bank customer attributes.
- Predicts subscription likelihood using a trained CatBoost model.
- Displays a performance leaderboard comparing evaluated models.
- Shows feature importance, SHAP plot, ROC curve, and confusion matrix visualizations.

## Installation

1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install required packages.

```powershell
pip install streamlit pandas numpy joblib pillow scikit-learn catboost
```

## Running the App

From the project root, run:

```powershell
streamlit run App\app.py
```

Then open the local Streamlit URL displayed in your terminal.

## Usage

1. Fill in the customer details in the app form.
2. Click **Predict**.
3. Review the prediction outcome and probability.
4. Explore model performance and visualization tabs.

## Notes

- The app loads the trained model from `Model/catboost_pipeline.pkl`.
- It creates additional features such as `contacted_before`, `total_loans`, and `age_balance` before prediction.

## License

This project is provided for demonstration and educational purposes.

## Author

Nisha M
