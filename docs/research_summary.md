# Survey of Existing Research and Available Solutions

## Capstone Topic

**Bank Customer Churn Prediction Using Machine Learning**

Customer churn prediction is a widely studied machine learning problem that focuses on identifying customers who are likely to discontinue using a product or service. Organizations use churn prediction models to proactively identify at-risk customers and implement retention strategies before revenue is lost. This research survey reviews existing literature and publicly available implementations to establish a baseline approach for this capstone project.

---

## Research Source 1: Customer Churn Prediction Model using Explainable Machine Learning

**Link:** https://arxiv.org/abs/2303.00960

This paper focuses on predicting customer churn using tree-based machine learning models while emphasizing the importance of model explainability. The authors compare several machine learning algorithms and identify XGBoost as one of the strongest-performing models for churn prediction. In addition to predictive performance, the paper uses SHAP values to explain which features contribute most to individual predictions. This research is relevant to my capstone because it demonstrates that accurate predictions should also be interpretable for business users who need to understand why customers are identified as being at risk of churn.

---

## Research Source 2: Research on Customer Churn Prediction and Model Interpretability Analysis

**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10707658/

This research evaluates multiple classification algorithms for customer churn prediction while emphasizing model interpretability. The authors compare different machine learning techniques and conclude that tree-based ensemble methods such as XGBoost provide strong predictive performance while remaining explainable through feature importance analysis. The research reinforces the importance of balancing predictive accuracy with transparency, making it well aligned with the goals of this capstone project.

---

## Research Source 3: Public Churn Prediction Implementation Using Logistic Regression, Random Forest, and XGBoost

**Link:** https://github.com/yasirsaleem502/Predict-Customer-Churn-Analysis-using-Logistic-Regression-Random-Forest-XG-boost

This public GitHub implementation demonstrates customer churn prediction using Python, Jupyter Notebook, pandas, scikit-learn, Logistic Regression, Random Forest, and XGBoost. The repository includes data preprocessing, exploratory data analysis, model training, evaluation, and comparison of multiple machine learning algorithms. This implementation was selected because it closely matches the workflow planned for this capstone project and provides a practical baseline that can be reproduced and expanded upon.

---

## Available Solution Selected for Reproduction

**Selected Implementation:** Predict Customer Churn Analysis using Logistic Regression, Random Forest, and XGBoost

**Source:** https://github.com/yasirsaleem502/Predict-Customer-Churn-Analysis-using-Logistic-Regression-Random-Forest-XG-boost

Rather than copying the project directly, I will reproduce the overall machine learning workflow demonstrated in the selected implementation while developing my own solution independently. The reproduction will include the following steps:

1. Load the Bank Customer Churn dataset.
2. Explore and inspect the dataset.
3. Clean and preprocess the data.
4. Encode categorical variables and prepare numerical features.
5. Split the dataset into training and testing sets.
6. Train baseline machine learning models.
7. Evaluate model performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
8. Compare model performance and identify the strongest baseline approach.

---

## Dataset Used for My Capstone

The dataset selected for this capstone is the **Bank Customer Churn (Churn Modelling) Dataset** obtained from Kaggle. It contains **10,000 customer records** and includes demographic and financial attributes such as credit score, geography, gender, age, tenure, account balance, and estimated salary. The target variable, **Exited**, indicates whether a customer left the bank.

Although this dataset represents the banking industry, the machine learning techniques used to predict customer churn are broadly applicable to many subscription-based and service-oriented businesses. The dataset provides a well-established benchmark for supervised binary classification and is widely used in machine learning education and research.

---

## What I Learned

The research consistently shows that traditional machine learning algorithms remain highly effective for structured customer churn datasets. Logistic Regression provides a simple and interpretable baseline model, while ensemble methods such as Random Forest and XGBoost generally achieve stronger predictive performance by capturing nonlinear relationships within the data.

Another important finding is that model interpretability is essential in business applications. A highly accurate churn prediction model provides greater business value when decision-makers can understand the factors influencing each prediction. Explainability techniques such as feature importance and SHAP values help transform machine learning models from "black boxes" into practical decision-support tools.

The research also highlights the importance of data preparation. Proper handling of missing values, encoding categorical variables, feature engineering, and addressing class imbalance can significantly improve model performance. These findings reinforce that building a successful machine learning solution requires a complete pipeline rather than focusing only on model selection.

---

## How My Capstone Will Improve or Extend Existing Work

The research and public implementations reviewed provide strong examples of customer churn prediction using traditional machine learning techniques. However, many publicly available projects conclude after reporting model evaluation metrics.

My capstone will extend these approaches by documenting the complete machine learning engineering workflow, including data preprocessing, exploratory data analysis, feature engineering, model comparison, performance evaluation, and deployment. The final project will package the trained model into a lightweight web application or REST API capable of generating churn predictions for new customer records.

In addition to predictive performance, the project will emphasize model transparency through feature importance analysis so that users can better understand the factors influencing churn predictions. By combining reproducible machine learning methods with practical deployment and documentation, the project aims to demonstrate both machine learning and software engineering best practices.
