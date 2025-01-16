import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# بارگذاری دیتاست Iris
iris = load_iris()

# تبدیل به DataFrame برای نمایش بهتر
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# نمایش نمونه‌ای از داده‌ها
print(df.head())

# نمایش اطلاعات کلی
print(df.info())

# انتخاب ویژگی‌ها (X) و هدف (y)
X = df.iloc[:, :-1]  # ویژگی‌ها (4 ستون اول)
y = df['species']    # هدف (ستون گونه‌ها)

# تقسیم داده‌ها به مجموعه آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ایجاد مدل جنگل تصادفی
model = RandomForestClassifier(n_estimators=100, random_state=42)

# آموزش مدل
model.fit(X_train, y_train)

# پیش‌بینی روی داده‌های آزمون
y_pred = model.predict(X_test)

# ماتریس سردرگمی
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# نمایش گزارش طبقه‌بندی
print("Classification Report:")
print(classification_report(y_test, y_pred))

new_data = [[5.1, 3.5, 1.4, 0.2]]  # ویژگی‌های یک گل
prediction = model.predict(new_data)
print("Predicted Species:", iris.target_names[prediction[0]])

# رسم ماتریس سردرگمی
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix_Forest classifier")
plt.show()

### logistic regression
from sklearn.linear_model import LogisticRegression

# ایجاد مدل Logistic Regression
model_lr = LogisticRegression(max_iter=200, random_state=42)

# آموزش مدل
model_lr.fit(X_train, y_train)

# پیش‌بینی روی داده‌های آزمون
y_pred_lr = model_lr.predict(X_test)

# ارزیابی مدل
print("Logistic Regression - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\nLogistic Regression - Classification Report:")
print(classification_report(y_test, y_pred_lr))
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, cmap="Blues", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix_logistic regression")
plt.show()

#SVM

from sklearn.svm import SVC

# ایجاد مدل SVM
model_svm = SVC(kernel='linear', random_state=42)  # می‌توانید kernel را به 'rbf' یا 'poly' تغییر دهید

# آموزش مدل
model_svm.fit(X_train, y_train)

# پیش‌بینی روی داده‌های آزمون
y_pred_svm = model_svm.predict(X_test)

# ارزیابی مدل
print("SVM - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

print("\nSVM - Classification Report:")
print(classification_report(y_test, y_pred_svm))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, cmap="Blues", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix_SVM")
plt.show()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_svm, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

