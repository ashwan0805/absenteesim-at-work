🏢 Absenteeism at Work - Machine Learning Analysis
This project focuses on predicting employee absenteeism hours using machine learning models (Random Forest and Decision Tree). The analysis is based on the UCI Absenteeism at Work Dataset, and helps identify key factors influencing employee absences.

📁 Files Included
absenteesm at work.py: Python script for data preprocessing, model training, evaluation, and visualization.

Absenteeism_at_work.csv: Dataset containing employee absenteeism records.

Output plots (attached):

Feature Importances from Random Forest

Actual vs Predicted Absenteeism Hours

📊 Dataset
The dataset contains information about 740 instances of employee absenteeism at a courier company, with features such as:

Reason for absence

Work load average per day

Day of the week

Seasons, transportation expense, age, BMI, and more

Target Variable: Absenteeism time in hours

🧠 Models Used
Decision Tree Regressor

Random Forest Regressor

Both models were trained to predict absenteeism hours based on the available features.

⚙️ Techniques Applied
Data Preprocessing using Pandas

Train-test split using train_test_split (80/20)

Model training with RandomForestRegressor and DecisionTreeRegressor

Evaluation using:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

Feature importance visualization

Actual vs Predicted scatter plot comparison

📈 Results
Model	MAE	MSE
Decision Tree	7.02	380.50
Random Forest	5.39	147.28

✅ Random Forest performed better than the Decision Tree in terms of prediction accuracy.

📌 Key Feature Insights
According to the Random Forest model, the most important features influencing absenteeism were:

Reason for absence

Work load average/day

Day of the week

Age

(See the feature importance bar chart in the output images.)

📷 Output Visualizations
1. Feature Importances from Random Forest

2. Actual vs Predicted Absenteeism Hours

📚 References
UCI Machine Learning Repository - Absenteeism at Work Dataset
