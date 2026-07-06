# Survey of Existing Research and Available Solutions

## Capstone Topic
Customer churn prediction using machine learning.

## Research Source 1: Customer Churn Prediction Model using Explainable Machine Learning
Link: https://arxiv.org/abs/2303.00960

This paper focuses on predicting customer churn using tree-based machine learning models and emphasizes the importance of model explainability. The authors compare several approaches and identify XGBoost as a strong-performing model for churn prediction. The paper also uses SHAP values to explain which features contribute most to churn predictions. This is relevant to my capstone because my project also aims to predict churn while making the model output understandable to business users.

## Research Source 2: Research on Customer Churn Prediction and Model Interpretability Analysis
Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10707658/

This research studies customer churn prediction using multiple classification models and focuses on model interpretability. It uses XGBoost and explanation methods to identify important churn drivers. This supports the direction of my capstone because it shows that churn prediction should not only optimize accuracy but also help users understand the reasons behind a prediction.

## Research Source 3: Customer Churn Prediction GitHub Topic
Link: https://github.com/topics/customer-churn-prediction

The GitHub topic page contains many public churn prediction projects using common tools such as pandas, scikit-learn, Logistic Regression, Random Forest, XGBoost, and visualization libraries. These projects provide available solution patterns that can be reproduced and adapted for my capstone.

## Available Solution Selected for Reproduction
Selected implementation type: a churn prediction notebook using Python, pandas, scikit-learn, Logistic Regression, Random Forest, and evaluation metrics.

Instead of copying a project directly, I reproduced the common solution approach used across public churn prediction notebooks:
1. Load a public churn dataset.
2. Clean and preprocess the data.
3. Encode categorical variables.
4. Split the data into training and testing sets.
5. Train baseline classification models.
6. Evaluate the models using accuracy, precision, recall, F1-score, and ROC-AUC.
7. Compare results and identify the best baseline approach.

## What I Learned
Existing churn prediction work commonly uses traditional machine learning models because churn datasets are usually structured tabular datasets. Logistic Regression is useful as a baseline because it is simple and interpretable. Tree-based models such as Random Forest and XGBoost often perform better because they can capture nonlinear relationships between customer features and churn behavior.

A major lesson from the research is that interpretability matters in business-facing churn prediction systems. A high-performing model is less useful if customer success teams cannot understand why a customer is considered at risk. For that reason, my capstone will include model evaluation and interpretability through feature importance or SHAP-style explanations if time permits.

## How My Capstone Will Improve or Extend Existing Work
Many public churn prediction notebooks stop after reporting model accuracy. My capstone will go further by packaging the model as a practical business tool. The final project will include a documented machine learning pipeline, model comparison, explainability, and a lightweight deployed application or API that can return churn predictions for new customer records.
