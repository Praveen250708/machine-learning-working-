import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris 
iris=load_iris()
X = iris.data 
y = iris.target 
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y 
print("First 5 rows of the Iris dataset:")
print(df.head())
print("\nDataset Information:")
df.info()
print("\nDescriptive Statistics:")
print(df.describe())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")   
print(f"Shape of y_train: {y_train.shape}") 
print(f"Shape of y_test: {y_test.shape}")    
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Gaussian Naive Bayes model trained successfully!")
y_pred = gnb.predict(X_test)
print("Actual labels (first 10):", y_test[:10])
print("Predicted labels (first 10):", y_pred[:10])
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}") 
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6)) 
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names) 
plt.ylabel('True Label')      
plt.title('Confusion Matrix for Gaussian Naive Bayes') 
plt.show() 
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:")
print(class_report)