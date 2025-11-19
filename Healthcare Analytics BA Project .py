import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle


df = pd.read_csv("Diabetes_Prediction_Project_Dataset.csv")


df.head()


print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())


df.describe()


sns.countplot(x='Outcome', data=df, palette='coolwarm')
plt.title("Distribution of Diabetes Outcome (0 = No, 1 = Yes)")
plt.show()


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.title("Feature Correlation Heatmap")
plt.show()


features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
plt.figure(figsize=(12,8))
for i, feature in enumerate(features, 1):
    plt.subplot(2,2,i)
    sns.histplot(df[feature], kde=True, bins=20, color='teal')
    plt.title(f"{feature} Distribution")
plt.tight_layout()
plt.show()


cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)


print("Missing Values After Replacing Zeros:\n", df.isnull().sum())

for col in cols_with_zeros:
    df[col].fillna(df[col].median(), inplace=True)


df.isnull().sum()


X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training and testing data prepared successfully!")


lr = LogisticRegression(max_iter=1000, solver='liblinear')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("✅ Models trained successfully!")


y_pred_lr = lr.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)

y_prob_lr = lr.predict_proba(X_test_scaled)[:,1]
y_prob_rf = rf.predict_proba(X_test)[:,1]


print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(12,5))
ConfusionMatrixDisplay.from_estimator(lr, X_test_scaled, y_test, cmap='Blues', values_format='d')
plt.title("Logistic Regression Confusion Matrix")


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, cmap='Greens', values_format='d')
plt.title("Random Forest Confusion Matrix")
plt.show()


fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


importance = rf.feature_importances_
fi = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=fi, palette='viridis')
plt.title("Feature Importance (Random Forest)")
plt.show()

fi

new_patient = [[2, 120, 70, 25, 80, 32.0, 0.45, 29]]

scaled_input = scaler.transform(new_patient)

prediction = lr.predict(scaled_input)
probability = lr.predict_proba(scaled_input)[0][1]


print("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
print("Probability of Diabetes:", round(probability, 2))
