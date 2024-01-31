import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Assuming x_train, x_test, y_train, y_test are your training and testing data
# x_train and x_test are assumed to be NumPy arrays, and y_train and y_test are 1D arrays or lists
x_train = np.load('./data/train_bovw_data.npy')

print(x_train.shape)

ones = [1] * 50
ones = np.array(ones)
ones = ones.reshape(-1, 1)
twos = np.array([2] * 50).T
twos = twos.reshape(-1, 1)
threes = np.array([3] * 50).T
threes = threes.reshape(-1, 1)

y_train = np.vstack((ones, twos, threes)).ravel()


x_test = np.load('./data/test_bovw_data.npy')
print(x_test.shape)

ones = [1] * 50
ones = np.array(ones)
ones = ones.reshape(-1, 1)
twos = np.array([2] * 50).T
twos = twos.reshape(-1, 1)
threes = np.array([3] * 50).T
threes = threes.reshape(-1, 1)

y_test = np.vstack((ones, twos, threes)).ravel()

n_neighbors=[1,3,5,7,9]
# Step 1: Create a KNN classifier

for k in n_neighbors:
    knn_classifier = KNeighborsClassifier(k)  # You can adjust the number of neighbors as needed
    
    # Step 2: Fit the classifier with the training data
    knn_classifier.fit(x_train, y_train)
    
    # Step 3: Make predictions on the test data
    y_pred = knn_classifier.predict(x_test)
    
    print("for k-nearest neighbours = ", k, "the specifics are: ")
    # Step 4: Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Accuracy:", accuracy)
    
    # Step 5: Calculate precision, recall, and F1-score for each class
    report = classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3'], output_dict=True)
    print("Classification Report:")
    for class_name in ['Class 1', 'Class 2', 'Class 3']:
        print(f"{class_name}:\n"
              f"  Precision: {report[class_name]['precision']:.4f}\n"
              f"  Recall:    {report[class_name]['recall']:.4f}\n"
              f"  F1-Score:  {report[class_name]['f1-score']:.4f}\n"
              f"  Support:   {report[class_name]['support']}\n")
    
    # Step 6: Calculate mean precision, recall, and F1-score
    mean_precision = np.mean([report[class_name]['precision'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
    mean_recall = np.mean([report[class_name]['recall'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
    mean_f1_score = np.mean([report[class_name]['f1-score'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
    
    print("Mean Precision:", mean_precision)
    print("Mean Recall:", mean_recall)
    print("Mean F1-Score:", mean_f1_score)
    
    # Step 7: Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    
    # Step 8: Display the confusion matrix using seaborn heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'],
                yticklabels=['Class 1', 'Class 2', 'Class 3'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have x_train, x_test, y_train, y_test

# Split the data into training and testing sets if not done already
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train = np.load('./data/train_bovw_data.npy')

ones = [1] * 50
ones = np.array(ones)
ones = ones.reshape(-1, 1)
twos = np.array([2] * 50).T
twos = twos.reshape(-1, 1)
threes = np.array([3] * 50).T
threes = threes.reshape(-1, 1)

y_train = np.vstack((ones, twos, threes)).ravel()


x_test = np.load('./data/test_bovw_data.npy')

ones = [1] * 50
ones = np.array(ones)
ones = ones.reshape(-1, 1)
twos = np.array([2] * 50).T
twos = twos.reshape(-1, 1)
threes = np.array([3] * 50).T
threes = threes.reshape(-1, 1)

y_test = np.vstack((ones, twos, threes)).ravel()
# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

print("\n LOGISTIC REGRESSION\n")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)

# Step 5: Calculate precision, recall, and F1-score for each class
report = classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3'], output_dict=True, zero_division=0)
print("Classification Report:")
for class_name in ['Class 1', 'Class 2', 'Class 3']:
    print(f"{class_name}:\n"
          f"  Precision: {report[class_name]['precision']:.4f}\n"
          f"  Recall:    {report[class_name]['recall']:.4f}\n"
          f"  F1-Score:  {report[class_name]['f1-score']:.4f}\n"
          f"  Support:   {report[class_name]['support']}\n")

# Step 6: Calculate mean precision, recall, and F1-score
mean_precision = np.mean([report[class_name]['precision'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
mean_recall = np.mean([report[class_name]['recall'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
mean_f1_score = np.mean([report[class_name]['f1-score'] for class_name in ['Class 1', 'Class 2', 'Class 3']])

print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean F1-Score:", mean_f1_score)

# Step 7: Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Step 8: Display the confusion matrix using seaborn heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'],
            yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()