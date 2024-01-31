import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions

# Load your datasets for each class
class1_data = pd.read_csv('./Group18/LS_Group18/Class1.txt', header=None, delimiter=' ')
class2_data = pd.read_csv('./Group18/LS_Group18/Class2.txt', header=None, delimiter=' ')
class3_data = pd.read_csv('./Group18/LS_Group18/Class3.txt', header=None, delimiter=' ')

# Assuming your class labels are 1, 2, and 3 for the three classes
class1_data['label'] = 1
class2_data['label'] = 2
class3_data['label'] = 3

# Concatenate the datasets
all_data = pd.concat([class1_data, class2_data, class3_data], ignore_index=True)

# Separate features and labels
X = all_data.iloc[:, :-1].values
Y = all_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Evaluation Metrics
accuracy = accuracy_score(Y_Test, Y_Pred)
precision_per_class = precision_score(Y_Test, Y_Pred, average=None)
mean_precision = precision_score(Y_Test, Y_Pred, average='micro')
recall_per_class = recall_score(Y_Test, Y_Pred, average=None)
mean_recall = recall_score(Y_Test, Y_Pred, average='micro')
f1_per_class = f1_score(Y_Test, Y_Pred, average=None)
mean_f1 = f1_score(Y_Test, Y_Pred, average='micro')

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Print Metrics
print(f"Accuracy: {accuracy}")
print(f"Precision per class: {precision_per_class}")
print(f"Mean Precision: {mean_precision}")
print(f"Recall per class: {recall_per_class}")
print(f"Mean Recall: {mean_recall}")
print(f"F1 Score per class: {f1_per_class}")
print(f"Mean F1 Score: {mean_f1}")
print(f"Confusion Matrix:\n{cm}")

# Decision Region Plot
plot_decision_regions(X_Test, Y_Test, clf=classifier, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions for SVM with Linear Kernel')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions

# Load your datasets for each class
class1_data = pd.read_csv('./Group18/LS_Group18/Class1.txt', header=None, delimiter=' ')
class2_data = pd.read_csv('./Group18/LS_Group18/Class2.txt', header=None, delimiter=' ')
class3_data = pd.read_csv('./Group18/LS_Group18/Class3.txt', header=None, delimiter=' ')

# Assuming your class labels are 1, 2, and 3 for the three classes
class1_data['label'] = 1
class2_data['label'] = 2
class3_data['label'] = 3

# Concatenate the datasets
all_data = pd.concat([class1_data, class2_data, class3_data], ignore_index=True)

# Separate features and labels
X = all_data.iloc[:, :-1].values
Y = all_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set with a polynomial kernel
# You can adjust the degree parameter for the polynomial degree
classifier = SVC(kernel='poly', degree=3, random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Evaluation Metrics
accuracy = accuracy_score(Y_Test, Y_Pred)
precision_per_class = precision_score(Y_Test, Y_Pred, average=None)
mean_precision = precision_score(Y_Test, Y_Pred, average='micro')
recall_per_class = recall_score(Y_Test, Y_Pred, average=None)
mean_recall = recall_score(Y_Test, Y_Pred, average='micro')
f1_per_class = f1_score(Y_Test, Y_Pred, average=None)
mean_f1 = f1_score(Y_Test, Y_Pred, average='micro')

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Print Metrics
print(f"Accuracy: {accuracy}")
print(f"Precision per class: {precision_per_class}")
print(f"Mean Precision: {mean_precision}")
print(f"Recall per class: {recall_per_class}")
print(f"Mean Recall: {mean_recall}")
print(f"F1 Score per class: {f1_per_class}")
print(f"Mean F1 Score: {mean_f1}")
print(f"Confusion Matrix:\n{cm}")

# Decision Region Plot
plot_decision_regions(X_Test, Y_Test, clf=classifier, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions for SVM with Polynomial Kernel')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions

# Load your datasets for each class
class1_data = pd.read_csv('./Group18/LS_Group18/Class1.txt', header=None, delimiter=' ')
class2_data = pd.read_csv('./Group18/LS_Group18/Class2.txt', header=None, delimiter=' ')
class3_data = pd.read_csv('./Group18/LS_Group18/Class3.txt', header=None, delimiter=' ')

# Assuming your class labels are 1, 2, and 3 for the three classes
class1_data['label'] = 1
class2_data['label'] = 2
class3_data['label'] = 3

# Concatenate the datasets
all_data = pd.concat([class1_data, class2_data, class3_data], ignore_index=True)

# Separate features and labels
X = all_data.iloc[:, :-1].values
Y = all_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set with an RBF kernel
# You can adjust the gamma parameter according to your needs
classifier = SVC(kernel='rbf', gamma='scale', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Evaluation Metrics
accuracy = accuracy_score(Y_Test, Y_Pred)
precision_per_class = precision_score(Y_Test, Y_Pred, average=None)
mean_precision = precision_score(Y_Test, Y_Pred, average='micro')
recall_per_class = recall_score(Y_Test, Y_Pred, average=None)
mean_recall = recall_score(Y_Test, Y_Pred, average='micro')
f1_per_class = f1_score(Y_Test, Y_Pred, average=None)
mean_f1 = f1_score(Y_Test, Y_Pred, average='micro')

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Print Metrics
print(f"Accuracy: {accuracy}")
print(f"Precision per class: {precision_per_class}")
print(f"Mean Precision: {mean_precision}")
print(f"Recall per class: {recall_per_class}")
print(f"Mean Recall: {mean_recall}")
print(f"F1 Score per class: {f1_per_class}")
print(f"Mean F1 Score: {mean_f1}")
print(f"Confusion Matrix:\n{cm}")

# Decision Region Plot
plot_decision_regions(X_Test, Y_Test, clf=classifier, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions for SVM with RBF Kernel')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions

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

# Convert the lists to pandas dataframes if needed
class1_data = pd.DataFrame(data1)
class2_data = pd.DataFrame(data2)
class3_data = pd.DataFrame(data3)

# Assuming your class labels are 1, 2, and 3 for the three classes
class1_data['label'] = 1
class2_data['label'] = 2
class3_data['label'] = 3

# Concatenate the datasets
all_data = pd.concat([class1_data, class2_data, class3_data], ignore_index=True)

# Separate features and labels
X = all_data.iloc[:, :-1].values
Y = all_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Evaluation Metrics
accuracy = accuracy_score(Y_Test, Y_Pred)
precision_per_class = precision_score(Y_Test, Y_Pred, average=None)
mean_precision = precision_score(Y_Test, Y_Pred, average='micro')
recall_per_class = recall_score(Y_Test, Y_Pred, average=None)
mean_recall = recall_score(Y_Test, Y_Pred, average='micro')
f1_per_class = f1_score(Y_Test, Y_Pred, average=None)
mean_f1 = f1_score(Y_Test, Y_Pred, average='micro')

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Print Metrics
print(f"Accuracy: {accuracy}")
print(f"Precision per class: {precision_per_class}")
print(f"Mean Precision: {mean_precision}")
print(f"Recall per class: {recall_per_class}")
print(f"Mean Recall: {mean_recall}")
print(f"F1 Score per class: {f1_per_class}")
print(f"Mean F1 Score: {mean_f1}")
print(f"Confusion Matrix:\n{cm}")

# Decision Region Plot
plot_decision_regions(X_Test, Y_Test, clf=classifier, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions for SVM with Linear Kernel')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions

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

# Convert the lists to pandas dataframes if needed
class1_data = pd.DataFrame(data1)
class2_data = pd.DataFrame(data2)
class3_data = pd.DataFrame(data3)

# Assuming your class labels are 1, 2, and 3 for the three classes
class1_data['label'] = 1
class2_data['label'] = 2
class3_data['label'] = 3

# Concatenate the datasets
all_data = pd.concat([class1_data, class2_data, class3_data], ignore_index=True)

# Separate features and labels
X = all_data.iloc[:, :-1].values
Y = all_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set with a polynomial kernel
# You can adjust the degree parameter for the polynomial degree
classifier = SVC(kernel='poly', degree=3, random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Evaluation Metrics
accuracy = accuracy_score(Y_Test, Y_Pred)
precision_per_class = precision_score(Y_Test, Y_Pred, average=None)
mean_precision = precision_score(Y_Test, Y_Pred, average='micro')
recall_per_class = recall_score(Y_Test, Y_Pred, average=None)
mean_recall = recall_score(Y_Test, Y_Pred, average='micro')
f1_per_class = f1_score(Y_Test, Y_Pred, average=None)
mean_f1 = f1_score(Y_Test, Y_Pred, average='micro')

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Print Metrics
print(f"Accuracy: {accuracy}")
print(f"Precision per class: {precision_per_class}")
print(f"Mean Precision: {mean_precision}")
print(f"Recall per class: {recall_per_class}")
print(f"Mean Recall: {mean_recall}")
print(f"F1 Score per class: {f1_per_class}")
print(f"Mean F1 Score: {mean_f1}")
print(f"Confusion Matrix:\n{cm}")

# Decision Region Plot
plot_decision_regions(X_Test, Y_Test, clf=classifier, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions for SVM with Polynomial Kernel')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions

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

# Convert the lists to pandas dataframes if needed
class1_data = pd.DataFrame(data1)
class2_data = pd.DataFrame(data2)
class3_data = pd.DataFrame(data3)

# Assuming your class labels are 1, 2, and 3 for the three classes
class1_data['label'] = 1
class2_data['label'] = 2
class3_data['label'] = 3

# Concatenate the datasets
all_data = pd.concat([class1_data, class2_data, class3_data], ignore_index=True)

# Separate features and labels
X = all_data.iloc[:, :-1].values
Y = all_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set with an RBF kernel
# You can adjust the gamma parameter according to your needs
classifier = SVC(kernel='rbf', gamma='scale', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results
Y_Pred = classifier.predict(X_Test)

# Evaluation Metrics
accuracy = accuracy_score(Y_Test, Y_Pred)
precision_per_class = precision_score(Y_Test, Y_Pred, average=None)
mean_precision = precision_score(Y_Test, Y_Pred, average='micro')
recall_per_class = recall_score(Y_Test, Y_Pred, average=None)
mean_recall = recall_score(Y_Test, Y_Pred, average='micro')
f1_per_class = f1_score(Y_Test, Y_Pred, average=None)
mean_f1 = f1_score(Y_Test, Y_Pred, average='micro')

# Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Print Metrics
print(f"Accuracy: {accuracy}")
print(f"Precision per class: {precision_per_class}")
print(f"Mean Precision: {mean_precision}")
print(f"Recall per class: {recall_per_class}")
print(f"Mean Recall: {mean_recall}")
print(f"F1 Score per class: {f1_per_class}")
print(f"Mean F1 Score: {mean_f1}")
print(f"Confusion Matrix:\n{cm}")

# Decision Region Plot
plot_decision_regions(X_Test, Y_Test, clf=classifier, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions for SVM with RBF Kernel')
plt.show()

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you already have x_train, x_test, y_train, and y_test

# Split the data into training and testing sets if you haven't already
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

# Create an SVM classifier with a linear kernel
svm_classifier = SVC(kernel='linear')

# Fit the model on the training data
svm_classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(x_test)

# Calculate accuracy
print("\n SVM Linear Kernel\n")
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

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you already have x_train, x_test, y_train, and y_test

# Split the data into training and testing sets if you haven't already
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

# Create an SVM classifier with a polynomial kernel
degrees=[1,2,3,4,5]
for deg in degrees:
    svm_classifier = SVC(kernel='poly', degree=deg)  # You can adjust the degree parameter as needed
    
    # Fit the model on the training data
    svm_classifier.fit(x_train, y_train)
    
    # Make predictions on the test set
    y_pred = svm_classifier.predict(x_test)
    
    # Calculate accuracy
    print(f"\n SVM Polynomial Kernel, degree = {deg}\n")
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Accuracy:", accuracy)
    
    # Calculate precision, recall, and F1-score for each class
    report = classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3'], output_dict=True, zero_division=0)
    print("Classification Report:")
    for class_name in ['Class 1', 'Class 2', 'Class 3']:
        print(f"{class_name}:\n"
              f"  Precision: {report[class_name]['precision']:.4f}\n"
              f"  Recall:    {report[class_name]['recall']:.4f}\n"
              f"  F1-Score:  {report[class_name]['f1-score']:.4f}\n"
              f"  Support:   {report[class_name]['support']}\n")
    
    # Calculate mean precision, recall, and F1-score
    mean_precision = np.mean([report[class_name]['precision'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
    mean_recall = np.mean([report[class_name]['recall'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
    mean_f1_score = np.mean([report[class_name]['f1-score'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
    
    print("Mean Precision:", mean_precision)
    print("Mean Recall:", mean_recall)
    print("Mean F1-Score:", mean_f1_score)
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    
    # Display the confusion matrix using seaborn heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'],
                yticklabels=['Class 1', 'Class 2', 'Class 3'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


    import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you already have x_train, x_test, y_train, and y_test

# Split the data into training and testing sets if you haven't already
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

degrees=[1,3,5]
gema = [0.001,0.01,0.5,0.1]
# Create an SVM classifier with a radial basis function (RBF) kernel
for deg in degrees:
    for gam in gema:
        svm_classifier = SVC(kernel='rbf', degree=deg, gamma = gam)
        
        # Fit the model on the training data
        svm_classifier.fit(x_train, y_train)
        
        # Make predictions on the test set
        y_pred = svm_classifier.predict(x_test)
        
        # Calculate accuracy
        print(f"\n SVM RBF Kernel: degree = {deg}, gamma = {gam}, width = {1/gam}\n")
        accuracy = accuracy_score(y_test, y_pred)
        print("Classification Accuracy:", accuracy)
        
        # Calculate precision, recall, and F1-score for each class
        report = classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3'], output_dict=True, zero_division=0)
        print("Classification Report:")
        for class_name in ['Class 1', 'Class 2', 'Class 3']:
            print(f"{class_name}:\n"
                  f"  Precision: {report[class_name]['precision']:.4f}\n"
                  f"  Recall:    {report[class_name]['recall']:.4f}\n"
                  f"  F1-Score:  {report[class_name]['f1-score']:.4f}\n"
                  f"  Support:   {report[class_name]['support']}\n")
        
        # Calculate mean precision, recall, and F1-score
        mean_precision = np.mean([report[class_name]['precision'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
        mean_recall = np.mean([report[class_name]['recall'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
        mean_f1_score = np.mean([report[class_name]['f1-score'] for class_name in ['Class 1', 'Class 2', 'Class 3']])
        
        print("Mean Precision:", mean_precision)
        print("Mean Recall:", mean_recall)
        print("Mean F1-Score:", mean_f1_score)
        
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)
        
        # Display the confusion matrix using seaborn heatmap
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'],
                    yticklabels=['Class 1', 'Class 2', 'Class 3'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()