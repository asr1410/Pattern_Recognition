import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

np.random.seed(687)

def evaluate_classifier_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=1)
    mean_precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average=None, zero_division=1)
    mean_recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=1)
    mean_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    return accuracy, precision, mean_precision, recall, mean_recall, f1, mean_f1

# Load datasets
data_class1 = np.loadtxt("./Group18/LS_Group18/Class1.txt")
data_class2 = np.loadtxt("./Group18/LS_Group18/Class2.txt")
data_class3 = np.loadtxt("./Group18/LS_Group18/Class3.txt")

# Split each class into training and testing sets
X_train_class1, X_test_class1, y_train_class1, y_test_class1 = train_test_split(data_class1, np.zeros(data_class1.shape[0]), test_size=0.3)
X_train_class2, X_test_class2, y_train_class2, y_test_class2 = train_test_split(data_class2, 1 * np.ones(data_class2.shape[0]), test_size=0.3)
X_train_class3, X_test_class3, y_train_class3, y_test_class3 = train_test_split(data_class3, 2 * np.ones(data_class3.shape[0]), test_size=0.3)

# Combine the training and testing sets for each class
X_train = np.vstack((X_train_class1, X_train_class2, X_train_class3))
X_test = np.vstack((X_test_class1, X_test_class2, X_test_class3))
y_train = np.concatenate((y_train_class1, y_train_class2, y_train_class3))
y_test = np.concatenate((y_test_class1, y_test_class2, y_test_class3))

# Apply Fisher Linear Discriminant Analysis and reduce to 1D
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Unimodal Gaussian Bayes Classifier with FDA
gnb = GaussianNB()
gnb.fit(X_train_lda, y_train)
y_pred_gnb = gnb.predict(X_test_lda)

# Evaluate performance metrics
accuracy_gnb, precision_gnb, mean_precision_gnb, recall_gnb, mean_recall_gnb, f1_gnb, mean_f1_gnb = evaluate_classifier_performance(y_test, y_pred_gnb)

# Print metrics for Unimodal Gaussian Bayes Classifier
print("Metrics for Unimodal Gaussian Bayes Classifier:")
print("Accuracy:", accuracy_gnb)
print("Precision (Class-wise):", precision_gnb)
print("Mean Precision:", mean_precision_gnb)
print("Recall (Class-wise):", recall_gnb)
print("Mean Recall:", mean_recall_gnb)
print("F-measure (Class-wise):", f1_gnb)
print("Mean F-measure:", mean_f1_gnb)

# Confusion matrix for Unimodal Gaussian Bayes Classifier
conf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb, labels=np.unique(y_test))
print("\nConfusion Matrix for Unimodal Gaussian Bayes Classifier:")
print(conf_matrix_gnb)

# Plot decision boundaries for Unimodal Gaussian Bayes Classifier
gnb2d = GaussianNB()
gnb2d.fit(X_train, y_train)

h = .02  # step size in the mesh
x_min, x_max = X_train[:, 0].min() - 3, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = gnb2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision boundaries
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)

# Plot testing data points
for label in np.unique(y_test):
    plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], label=f'Train Data Class {int(label)}', marker='o', edgecolors='k')

plt.scatter(X_train_lda, np.zeros_like(X_train_lda), c=y_train, s=25, cmap='viridis', marker='o', edgecolors='k', label='1-D Train Data')
plt.title('Decision Boundaries of Unimodal Gaussian Bayes Classifier \nand 1-dimensional Reduced Representation of Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend()
plt.show()

np.random.seed(33)

# Gaussian Mixture Model Bayes Classifier with FDA
gmm = GaussianMixture(n_components=len(np.unique(y_train)))
gmm.fit(X_train_lda)
y_pred_gmm = gmm.predict(X_test_lda)

# Evaluate performance metrics
accuracy_gmm, precision_gmm, mean_precision_gmm, recall_gmm, mean_recall_gmm, f1_gmm, mean_f1_gmm = evaluate_classifier_performance(y_test, y_pred_gmm)

# Print metrics for Gaussian Mixture Model Bayes Classifier
print("\nMetrics for Gaussian Mixture Model Bayes Classifier:")
print("Accuracy:", accuracy_gmm)
print("Precision (Class-wise):", precision_gmm)
print("Mean Precision:", mean_precision_gmm)
print("Recall (Class-wise):", recall_gmm)
print("Mean Recall:", mean_recall_gmm)
print("F-measure (Class-wise):", f1_gmm)
print("Mean F-measure:", mean_f1_gmm)

# Confusion matrix for Gaussian Mixture Model Bayes Classifier
conf_matrix_gmm = confusion_matrix(y_test, y_pred_gmm, labels=np.unique(y_test))
print("\nConfusion Matrix for Gaussian Mixture Model Bayes Classifier:")
print(conf_matrix_gmm)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

np.random.seed(1)

def evaluate_classifier_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=1)
    mean_precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average=None, zero_division=1)
    mean_recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=1)
    mean_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    return accuracy, precision, mean_precision, recall, mean_recall, f1, mean_f1

class_0_data = []
class_1_data = []
class_2_data = []

with open('./Group18/NLS_Group18.txt', 'r') as file:
    # Skip the first line
    next(file)

    for i, line in enumerate(file):
        values = line.strip().split()

        # Convert values to float
        values = [float(val) for val in values]

        # Decide which class the entry belongs to based on its index
        if i < 500:
            class_0_data.append(values)
        elif i < 1000:
            class_1_data.append(values)
        else:
            class_2_data.append(values)

# Convert lists to NumPy arrays
data_class1 = np.array(class_0_data)
data_class2 = np.array(class_1_data)
data_class3 = np.array(class_2_data)

# Split each class into training and testing sets
X_train_class1, X_test_class1, y_train_class1, y_test_class1 = train_test_split(data_class1, np.zeros(data_class1.shape[0]), test_size=0.3)
X_train_class2, X_test_class2, y_train_class2, y_test_class2 = train_test_split(data_class2, 1 * np.ones(data_class2.shape[0]), test_size=0.3)
X_train_class3, X_test_class3, y_train_class3, y_test_class3 = train_test_split(data_class3, 2 * np.ones(data_class3.shape[0]), test_size=0.3)

# Combine the training and testing sets for each class
X_train = np.vstack((X_train_class1, X_train_class2, X_train_class3))
X_test = np.vstack((X_test_class1, X_test_class2, X_test_class3))
y_train = np.concatenate((y_train_class1, y_train_class2, y_train_class3))
y_test = np.concatenate((y_test_class1, y_test_class2, y_test_class3))

# Apply Fisher Linear Discriminant Analysis and reduce to 1D
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Unimodal Gaussian Bayes Classifier with FDA
gnb = GaussianNB()
gnb.fit(X_train_lda, y_train)
y_pred_gnb = gnb.predict(X_test_lda)

# Evaluate performance metrics
accuracy_gnb, precision_gnb, mean_precision_gnb, recall_gnb, mean_recall_gnb, f1_gnb, mean_f1_gnb = evaluate_classifier_performance(y_test, y_pred_gnb)

# Print metrics for Unimodal Gaussian Bayes Classifier
print("Metrics for Unimodal Gaussian Bayes Classifier:")
print("Accuracy:", accuracy_gnb)
print("Precision (Class-wise):", precision_gnb)
print("Mean Precision:", mean_precision_gnb)
print("Recall (Class-wise):", recall_gnb)
print("Mean Recall:", mean_recall_gnb)
print("F-measure (Class-wise):", f1_gnb)
print("Mean F-measure:", mean_f1_gnb)

# Confusion matrix for Unimodal Gaussian Bayes Classifier
conf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb, labels=np.unique(y_test))
print("\nConfusion Matrix for Unimodal Gaussian Bayes Classifier:")
print(conf_matrix_gnb)

# Plot decision boundaries for Unimodal Gaussian Bayes Classifier
gnb2d = GaussianNB()
gnb2d.fit(X_train, y_train)

h = .02  # step size in the mesh
x_min, x_max = X_train[:, 0].min() - 1.5, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = gnb2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision boundaries
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)

# Plot testing data points
for label in np.unique(y_test):
    plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], label=f'Train Data Class {int(label)}', marker='o', edgecolors='k')

plt.scatter(X_train_lda, np.zeros_like(X_train_lda), c=y_train, cmap='viridis', marker='o', edgecolors='k', label='1-D Train Data')
plt.title('Decision Boundaries of Unimodal Gaussian Bayes Classifier \nand 1-dimensional Reduced Representation of Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.legend()
plt.show()

np.random.seed(33)

# Gaussian Mixture Model Bayes Classifier with FDA
gmm = GaussianMixture(n_components=len(np.unique(y_train)))
gmm.fit(X_train_lda)
y_pred_gmm = gmm.predict(X_test_lda)

# Evaluate performance metrics
accuracy_gmm, precision_gmm, mean_precision_gmm, recall_gmm, mean_recall_gmm, f1_gmm, mean_f1_gmm = evaluate_classifier_performance(y_test, y_pred_gmm)

# Print metrics for Gaussian Mixture Model Bayes Classifier
print("\nMetrics for Gaussian Mixture Model Bayes Classifier:")
print("Accuracy:", accuracy_gmm)
print("Precision (Class-wise):", precision_gmm)
print("Mean Precision:", mean_precision_gmm)
print("Recall (Class-wise):", recall_gmm)
print("Mean Recall:", mean_recall_gmm)
print("F-measure (Class-wise):", f1_gmm)
print("Mean F-measure:", mean_f1_gmm)

# Confusion matrix for Gaussian Mixture Model Bayes Classifier
conf_matrix_gmm = confusion_matrix(y_test, y_pred_gmm, labels=np.unique(y_test))
print("\nConfusion Matrix for Gaussian Mixture Model Bayes Classifier:")
print(conf_matrix_gmm)