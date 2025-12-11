# AI HealthGuard

**AI HealthGuard** is a machine learning project designed to analyze health data and predict potential health outcomes. The project utilizes a structured pipeline starting from Exploratory Data Analysis (EDA) to training various machine learning models, culminating in a high-performance XGBoost model for final comparison and prediction.

## *****This project is still in development**

## 📂 Project Structure

```text
AI-healthguard/
├── EDA.py                         # Script for Exploratory Data Analysis and visualization
├── train.py                       # Script to train and evaluate baseline machine learning models
├── XGBoost.py                     # Script to train the XGBoost model and perform model comparison
├── processed.cleveland.data       # Dataset
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation
```

# Getting Started
## Prerequisites
Ensure you have Python installed (version 3.8 or higher is recommended). You will also need standard data science libraries.
Installation
1. Clone the repository:
```
git clone [https://github.com/VaradPawaskar/AI-healthguard.git](https://github.com/VaradPawaskar/AI-healthguard.git)
cd AI-healthguard
```
2. Install dependencies: It is recommended to use a virtual environment. You can install the required packages using the following command (assuming you create a requirements.txt with libraries like pandas, numpy, scikit-learn, xgboost, matplotlib, and seaborn):
```
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```
# Usage Workflow
To reproduce the analysis and model training, please execute the scripts in the following specific order:

- Step 1: Exploratory Data Analysis (EDA)
Run the EDA script to visualize the dataset, check for missing values, and understand correlations between features.
```
python EDA.py
```
- Output: Generates statistical summaries and visualization plots (histograms, heatmaps, etc.) to give insights into the data distribution.


- Step 2: Model Training
Run the training script to pre-process the data and train baseline machine learning models (e.g., Logistic Regression, Decision Trees, Random Forest).
```
python train.py
```
- Output: Trains models, prints preliminary evaluation metrics (Accuracy, Precision, Recall), and may save model artifacts.


- Step 3: XGBoost & Model Comparison
Run the XGBoost script to train the gradient boosting model and compare its performance against the models trained in Step 2.
```
python XGBoost.py
```
- Output: * Trains the XGBoost classifier.
  - Performs a comparative analysis of all models.
  - Displays/saves a comparison plot or table showing which model performs best.****

- Step 4: Running the Streamlit Dashboard
  Run the Streamlit Dashboard by entering the command
```
streamlit run dashboard.py
```
- Output: Creates a Streamlit Dashboard at localhost which can be viewed inside the browser.

