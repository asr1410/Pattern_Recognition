import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from itertools import combinations
from matplotlib.colors import ListedColormap
# Read data from text files
data1 = np.loadtxt('./Group18/LS_Group18/Class1.txt')
data2 = np.loadtxt('./Group18/LS_Group18/Class2.txt')
data3 = np.loadtxt('./Group18/LS_Group18/Class3.txt')

# Assign labels to the data
labels1 = np.ones(data1.shape[0])
labels2 = 2 * np.ones(data2.shape[0])
labels3 = 3 * np.ones(data3.shape[0])

# Concatenate the data and labels
X = np.concatenate((data1, data2, data3), axis=0)
y = np.concatenate((labels1, labels2, labels3), axis=0)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Perceptron-based classifier
clf = Perceptron(tol=1e-7, random_state=0)
clf.fit(X_train, y_train)

# Predict the test set results
y_pred = clf.predict(X_test)

# Print classification metrics
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

classes = np.unique(y_train)
colors = {1: ('red', 'lightcoral'), 2: ('blue', 'lightblue'), 3: ('purple', 'lavender')}

for class_1, class_2 in combinations(classes, 2):
    mask_train = np.isin(y_train, [class_1, class_2])
    X_train_binary = X_train[mask_train]
    y_train_binary = y_train[mask_train]

    clf = Perceptron(tol=1e-7, random_state=0)
    clf.fit(X_train_binary, y_train_binary)

    x_min, x_max = X_train_binary[:, 0].min() - 1, X_train_binary[:, 0].max() + 5
    y_min, y_max = X_train_binary[:, 1].min() - 1, X_train_binary[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap([colors[class_1][1], colors[class_2][1]]))

    plt.scatter(X_train_binary[:, 0], X_train_binary[:, 1], c=y_train_binary, cmap=ListedColormap([colors[class_1][0], colors[class_2][0]]), s=4)

    mask_test = np.isin(y_test, [class_1, class_2])
    X_test_binary = X_test[mask_test]
    y_test_binary = y_test[mask_test]

    y_pred = clf.predict(X_test_binary)

    mask_correct = y_test_binary == y_pred
    plt.scatter(X_test_binary[mask_correct, 0], X_test_binary[mask_correct, 1], c='black', s=4, label='Correctly classified')

    mask_incorrect = y_test_binary != y_pred
    plt.scatter(X_test_binary[mask_incorrect, 0], X_test_binary[mask_incorrect, 1], c='red', s=4, label='Misclassified')

    plt.title(f'Decision boundary between class {int(class_1)} and class {int(class_2)}')
    plt.legend()

plt.show()


from sklearn.multiclass import OneVsRestClassifier
region_colors = ['lightcoral', 'lightblue', 'lavender']
point_colors = ['red', 'blue', 'purple']

region_cmap = ListedColormap(region_colors)
point_cmap = ListedColormap(point_colors)

clf = OneVsRestClassifier(Perceptron(tol=1e-7, random_state=0))
clf.fit(X_train, y_train)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 5
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=region_cmap)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=point_cmap, s = 4)
y_pred = clf.predict(X_test)

mask_correct = y_test == y_pred
plt.scatter(X_test[mask_correct, 0], X_test[mask_correct, 1], c='black', s=4, label='Correctly classified')

mask_incorrect = y_test != y_pred
plt.scatter(X_test[mask_incorrect, 0], X_test[mask_incorrect, 1], c='yellow', s=4, label='Misclassified')
plt.title(f'Decision boundary between class all the three classes')

plt.legend()
plt.show()

data1 = []
data2 = []
data3 = []

with open('./Group18/NLS_Group18.txt', 'r') as file:
    # Skip the first line
    next(file)
    
    for line in file:
        values = list(map(float, line.split()))
        
        if len(data1) < 500:
            data1.append(values)
        elif len(data2) < 500:
            data2.append(values)
        else:
            data3.append(values)

# Convert lists to NumPy arrays
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.array(data3)

# Assign labels to the data
labels1 = np.ones(data1.shape[0])
labels2 = 2 * np.ones(data2.shape[0])
labels3 = 3 * np.ones(data3.shape[0])

# Concatenate the data and labels
X = np.concatenate((data1, data2, data3), axis=0)
y = np.concatenate((labels1, labels2, labels3), axis=0)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Perceptron-based classifier
clf = Perceptron(tol=1e-7, random_state=0)
clf.fit(X_train, y_train)

# Predict the test set results
y_pred = clf.predict(X_test)

# Print classification metrics
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

classes = np.unique(y_train)
colors = {1: ('red', 'lightcoral'), 2: ('blue', 'lightblue'), 3: ('purple', 'lavender')}

for class_1, class_2 in combinations(classes, 2):
    mask_train = np.isin(y_train, [class_1, class_2])
    X_train_binary = X_train[mask_train]
    y_train_binary = y_train[mask_train]

    clf = Perceptron(tol=1e-7, random_state=0)
    clf.fit(X_train_binary, y_train_binary)

    x_min, x_max = X_train_binary[:, 0].min() - 1, X_train_binary[:, 0].max() + 1
    y_min, y_max = X_train_binary[:, 1].min() - 1, X_train_binary[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap([colors[class_1][1], colors[class_2][1]]))

    plt.scatter(X_train_binary[:, 0], X_train_binary[:, 1], c=y_train_binary, cmap=ListedColormap([colors[class_1][0], colors[class_2][0]]), s=4)

    mask_test = np.isin(y_test, [class_1, class_2])
    X_test_binary = X_test[mask_test]
    y_test_binary = y_test[mask_test]

    y_pred = clf.predict(X_test_binary)

    mask_correct = y_test_binary == y_pred
    plt.scatter(X_test_binary[mask_correct, 0], X_test_binary[mask_correct, 1], c='black', s=4, label='Correctly classified')

    mask_incorrect = y_test_binary != y_pred
    plt.scatter(X_test_binary[mask_incorrect, 0], X_test_binary[mask_incorrect, 1], c='red', s=4, label='Misclassified')

    plt.title(f'Decision boundary between class {int(class_1)} and class {int(class_2)}')
    plt.legend()

plt.show()


from sklearn.multiclass import OneVsRestClassifier
region_colors = ['lightcoral', 'lightblue', 'lavender']
point_colors = ['red', 'blue', 'purple']

region_cmap = ListedColormap(region_colors)
point_cmap = ListedColormap(point_colors)

clf = OneVsRestClassifier(Perceptron(tol=1e-7, random_state=0))
clf.fit(X_train, y_train)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=region_cmap)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=point_cmap, s = 4)
y_pred = clf.predict(X_test)

mask_correct = y_test == y_pred
plt.scatter(X_test[mask_correct, 0], X_test[mask_correct, 1], c='black', s=4, label='Correctly classified')

mask_incorrect = y_test != y_pred
plt.scatter(X_test[mask_incorrect, 0], X_test[mask_incorrect, 1], c='yellow', s=4, label='Misclassified')
plt.title(f'Decision boundary between class all the three classes')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap

# Load Dataset-2 (BoVW) from separate files
X_train_bovw = np.load('./data/train_bovw_data.npy')  # Replace with your actual file path
y_train_bovw_str = np.load('./data/train_bovw_labels.npy')  # Replace with your actual file path
X_test_bovw = np.load('./data/test_bovw_data.npy')  # Replace with your actual file path
y_test_bovw_str = np.load('./data/test_bovw_labels.npy')  # Replace with your actual file path

# Encode labels
label_encoder = LabelEncoder()
y_train_bovw = label_encoder.fit_transform(y_train_bovw_str)
y_test_bovw = label_encoder.transform(y_test_bovw_str)

# Train Perceptron without PCA
clf_perceptron = Perceptron(tol=1e-7, random_state=0)
clf_perceptron.fit(X_train_bovw, y_train_bovw)

# Predictions on test data without PCA
y_pred_perceptron = clf_perceptron.predict(X_test_bovw)

# Print classification report
print("Test Classification Report:")
print(classification_report(y_test_bovw, y_pred_perceptron, target_names=label_encoder.classes_))

# Print confusion matrix
print("Test Confusion Matrix:")
print(confusion_matrix(y_test_bovw, y_pred_perceptron))

from sklearn.decomposition import PCA

# Apply PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
X_train_bovw_pca = pca.fit_transform(X_train_bovw)
X_test_bovw_pca = pca.transform(X_test_bovw)

# Train Perceptron with PCA
clf_perceptron_pca = Perceptron(tol=1e-7, random_state=0)
clf_perceptron_pca.fit(X_train_bovw_pca, y_train_bovw)

# Predictions on test data with PCA
y_pred_perceptron_pca = clf_perceptron_pca.predict(X_test_bovw_pca)

# Print classification report
print("Test Classification Report with PCA:")
print(classification_report(y_test_bovw, y_pred_perceptron_pca, target_names=label_encoder.classes_))

# Print confusion matrix
print("Test Confusion Matrix with PCA:")
print(confusion_matrix(y_test_bovw, y_pred_perceptron_pca))

# Calculate the mesh grid
x_min, x_max = X_train_bovw_pca[:, 0].min() , X_train_bovw_pca[:, 0].max()+0.1
y_min, y_max = X_train_bovw_pca[:, 1].min()-0.1, X_train_bovw_pca[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predictions on the 2D mesh grid
Z = clf_perceptron_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundary and data points
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['lightcoral', 'lightblue', 'lavender']))
# plt.scatter(X_train_bovw_pca[:, 0], X_train_bovw_pca[:, 1], c=y_train_bovw, cmap='viridis', edgecolors='k', marker='o', s=2, linewidth=1.5, label='Train Data')
plt.scatter(X_test_bovw_pca[:, 0], X_test_bovw_pca[:, 1], c=y_test_bovw, s=15, label='Test Data')

# Mark misclassifications on the test set
misclassified = y_test_bovw != y_pred_perceptron_pca
plt.scatter(X_test_bovw_pca[misclassified, 0], X_test_bovw_pca[misclassified, 1], c='yellow', s=15, label='Misclassified')

# Legend and labels
plt.legend()
plt.title('Decision Boundary for Perceptron with PCA (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()