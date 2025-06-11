# Automobile-Customer-Segmentation
ML-based customer segmentation model for an automobile company. Predicts business-defined segments (A, B, C, D) using LightGBM and engineered features. Includes Streamlit deployment. Also explores clustering, SHAP analysis, and model limitations from real-world data.
This project builds a machine learning pipeline to predict customer segments (A, B, C, D) for an automobile company based on demographic and behavioral features. The segmentation labels were defined by the company's internal sales team based on historical purchase behavior and marketing performance.
The goal was to recreate this segmentation logic using supervised learning, validate natural separability using clustering, and deploy a user-friendly Streamlit app for real-time customer prediction.
Despite achieving 62.1% accuracy with LightGBM, extensive analysis revealed that Segment D dominated predictions, suggesting the business-defined labels may not be clearly distinguishable from the input features alone. KMeans clustering yielded low silhouette scores, further confirming the absence of strong natural clusters.
The project highlights real-world challenges in machine learning, such as label ambiguity, feature overlap, and the need for business-context alignment. It demonstrates strong practices in data preprocessing, feature engineering, model tuning, class balancing, SHAP interpretation, and deployment.

Dataset Overview:
2,627 records of customer profiles
Features: Age, Gender, Marital Status, Profession, Spending Score, Family Size, etc.
Target: Segmentation labels (A, B, C, D) assigned by the company’s business process

Techniques Used:
 Exploratory Data Analysis (EDA)
 Feature Engineering (HighSpender, Senior, BigFamily, etc.)
 Label Encoding
 KMeans Clustering (Unsupervised)
 LightGBM Classifier (Supervised)
 Ensemble Modeling
 SHAP Analysis for Feature Importance
 Streamlit Deployment


Key Insights:
Clustering Didn’t Work Well
KMeans clustering yielded low silhouette scores (< 0.3), indicating no strong natural customer clusters.
Concluded that business-defined segments (A/B/C/D) don’t map cleanly to numerical features.
Supervised Models Performed Better
LightGBM classifier achieved 62.1% accuracy.
Best model trained with balanced class weights, engineered features, and tuning.

Important Observation
Segment D dominates predictions even for customers with clearly different profiles.
Why?
Company-assigned labels likely originate from internal rules, not clean data patterns.
Segment D includes diverse customers, making it broad and hard to differentiate from others.
SHAP and feature analysis showed overlap between segment profiles.


Final Model Performance (After Merging B & C)
Segment	Precision	Recall	F1-Score

A	0.43	0.53	0.47

BC	0.76	0.63	0.69

D	0.64	0.69	0.66

Overall Accuracy			62.1%

