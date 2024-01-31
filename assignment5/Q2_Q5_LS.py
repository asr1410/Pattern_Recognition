import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import math

def calculate_mean(arr):
    total = 0
    count = 0
    for element in arr:
        total += element
        count += 1
    if count == 0:
        return 0
    return total / count

def split_data(X, y, test_size=0.3, seed=None):
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def calculate_covariance_matrix(data):
    num_samples, num_features = data.shape
    mean_vector = np.mean(data, axis=0)
    centered_data = data - mean_vector
    covariance_matrix = np.zeros((num_features, num_features))
    for i in range(num_samples):
        for j in range(num_features):
            for k in range(num_features):
                covariance_matrix[j, k] += centered_data[i, j] * centered_data[i, k]

    covariance_matrix /= (num_samples - 1)
    return covariance_matrix

# Function to calculate covariance matrices for different cases
def calculate_covariance_matrices(data_list, means_list, case):
    x = data_list[0]
    y = data_list[1]
    z = data_list[2]

    if case == 1:
        A = np.cov(x.T)
        B = np.cov(y.T)
        C = np.cov(z.T)
        t = np.add(A, B)
        avg = (1/3) * np.add(t, C)
        var = np.diag(avg)
        avg_var = np.sum(var) / len(var)
        cov_mat = np.identity(len(avg)) * avg_var
        return cov_mat

    elif case == 2:
        A = np.cov(x.T)
        B = np.cov(y.T)
        C = np.cov(z.T)
        t = np.add(A, B)
        avg_cov_mat = (1/3) * np.add(t, C)

        return avg_cov_mat

    elif case == 3:
        A = np.cov(x.T)
        B = np.cov(y.T)
        C = np.cov(z.T)
        a = np.diag(A)
        b = np.diag(B)
        c = np.diag(C)
        cov_mat_x = np.identity(len(a)) * np.transpose(a)
        cov_mat_y = np.identity(len(b)) * np.transpose(b)
        cov_mat_z = np.identity(len(c)) * np.transpose(c)

        return cov_mat_x, cov_mat_y, cov_mat_z

    elif case == 4:
        A = np.cov(x.T)
        B = np.cov(y.T)
        C = np.cov(z.T)
        return A, B, C

    else:
        raise ValueError("Invalid case. Choose from 1, 2, 3, or 4.")

def calculate_log_likelihood_single(data_point, mean, cov_matrix, prior):
    centered_x = np.array(data_point) - np.array(mean)
    inv_cov_mat = np.linalg.inv(cov_matrix)
    det_cov_mat = np.linalg.det(cov_matrix)
    log_likelihood = (-1/2) * np.matmul(np.matmul(np.transpose(centered_x), inv_cov_mat) , centered_x) - (1/2) * math.log(det_cov_mat) + math.log(prior)
    return log_likelihood

# Function to load and preprocess data
def load_and_preprocess_data(file_path, class_label, color):
    data = np.genfromtxt(file_path, delimiter=' ')
    X, y = data[:, 0], data[:, 1]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, seed=42)

    mean_x = calculate_mean(X_train)
    mean_y = calculate_mean(y_train)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, c=color, label=f'Class {class_label} Training Data')
    plt.scatter(mean_x, mean_y, c='black', label=f'Class {class_label} Mean')
    plt.title(f'Class {class_label} Training Data')
    plt.legend()
    plt.show()

    return X_train, y_train, mean_x, mean_y

# Function to plot the decision boundaries
def plot_decision_boundaries(X_train_list, y_train_list, mean_x_list, mean_y_list, covariance_matrices):
    min_x = min(np.min(X_train_list), np.min(y_train_list))
    max_x = max(np.max(X_train_list), np.max(y_train_list))
    min_y = min(np.min(X_train_list), np.min(y_train_list))
    max_y = max(np.max(X_train_list), np.max(y_train_list))

    num_points = 500
    X_color = np.linspace(min_x - 5, max_x + 5, num_points)
    Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

    X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)
    class_labels = np.zeros(X_mesh.shape)

    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            x_val = X_mesh[i, j]
            y_val = Y_mesh[i, j]

            likelihoods = []
            for mean_x, mean_y, cov_matrix in zip(mean_x_list, mean_y_list, covariance_matrices):
                likelihood = calculate_log_likelihood_single([x_val, y_val], [mean_x, mean_y], cov_matrix)
                likelihoods.append(likelihood)

            max_likelihood_class = np.argmax(likelihoods)
            class_labels[i, j] = max_likelihood_class

    plt.figure(figsize=(10, 6))
    for X_train, y_train, color in zip(X_train_list, y_train_list, ['darkred', 'darkblue', 'darkgreen']):
        plt.scatter(X_train, y_train, c=color, label=f'Class {class_label} Training Data')
    plt.contourf(X_mesh, Y_mesh, class_labels, levels=[0, 0.5, 1.5, 2.5], colors=['r', 'g', 'y'], alpha=0.3)
    plt.legend()
    plt.show()

file_path_c1 = "./Group18/LS_Group18/Class1.txt"
data_c1 = np.genfromtxt(file_path_c1, delimiter=' ')
X_c1, y_c1 = data_c1[:, 0], data_c1[:, 1]

X_train_c1, X_test_c1, y_train_c1, y_test_c1 = split_data(X_c1, y_c1, test_size=0.3, seed=42)

mean_x_c1 = calculate_mean(X_train_c1)
mean_y_c1 = calculate_mean(y_train_c1)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Training Data')
plt.scatter(X_test_c1, y_test_c1, c='darkblue', label='Test Data')
plt.scatter(mean_x_c1, mean_y_c1, c='black', label='Mean')
plt.title('Class 1 Training Data and Test Data')
plt.legend()
plt.show()

file_path_c2 = "./Group18/LS_Group18/Class2.txt"
data_c2 = np.genfromtxt(file_path_c2, delimiter=' ')
X_c2, y_c2 = data_c2[:, 0], data_c2[:, 1]
X_train_c2, X_test_c2, y_train_c2, y_test_c2 = split_data(X_c2, y_c2, test_size=0.3, seed=42)

mean_x_c2 = calculate_mean(X_train_c2)
mean_y_c2 = calculate_mean(y_train_c2)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Training Data')
plt.scatter(mean_x_c2, mean_y_c2, c='black', label='Mean')
plt.title('Class 2 Training Data')
plt.legend()
plt.show()

file_path_c3 = "./Group18/LS_Group18/Class3.txt"
data_c3 = np.genfromtxt(file_path_c3, delimiter=' ')
X_c3, y_c3 = data_c3[:, 0], data_c3[:, 1]
X_train_c3, X_test_c3, y_train_c3, y_test_c3 = split_data(X_c3, y_c3, test_size=0.3, seed=42)

mean_x_c3 = calculate_mean(X_train_c3)
mean_y_c3 = calculate_mean(y_train_c3)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_c3, y_train_c3, c='darkgreen', label='Training Data')
plt.scatter(mean_x_c3, mean_y_c3, c='black', label='Mean')
plt.title('Class 3 Training Data')
plt.legend()
plt.show()

lenC1 = len(X_c1)
lenC2 = len(X_c2)
lenC3 = len(X_c3)
total =lenC1 + lenC2 + lenC3
print(total)
prior1 = float(lenC1/total)
prior2 = float(lenC2/total)
prior3 = float(lenC3/total)

plt.figure(figsize=(10, 6))

plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')
plt.scatter(mean_x_c1, mean_y_c1, c='black', label='Class 1 Mean')

plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')
plt.scatter(mean_x_c2, mean_y_c2, c='black', label='Class 2 Mean')

plt.scatter(X_train_c3, y_train_c3, c='darkgreen', label='Class 3 Training Data')
plt.scatter(mean_x_c3, mean_y_c3, c='black', label='Class 3 Mean')

plt.title('Training Data and Mean for Three Classes')
plt.legend()
plt.show()


lenC1 = len(X_c1)
lenC2 = len(X_c2)
lenC3 = len(X_c3)
total =lenC1 + lenC2 + lenC3
print(total)
prior1 = float(lenC1/total)
prior2 = float(lenC2/total)
prior3 = float(lenC3/total)

plt.figure(figsize=(10, 6))

plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')
plt.scatter(mean_x_c1, mean_y_c1, c='black', label='Class 1 Mean')

plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')
plt.scatter(mean_x_c2, mean_y_c2, c='black', label='Class 2 Mean')

plt.scatter(X_train_c3, y_train_c3, c='darkgreen', label='Class 3 Training Data')
plt.scatter(mean_x_c3, mean_y_c3, c='black', label='Class 3 Mean')


plt.scatter(X_test_c1, y_test_c1, c='magenta', label='Test Data C1')
plt.scatter(X_test_c2, y_test_c2, c='yellow', label='Test Data C2')
plt.scatter(X_test_c3, y_test_c3, c='pink', label='Test Data C3')


plt.title('Training Data and Mean for Three Classes')
plt.legend()
plt.show()



# Data and means lists for different cases
data_list = [
    np.column_stack((X_train_c1, y_train_c1)),
    np.column_stack((X_train_c2, y_train_c2)),
    np.column_stack((X_train_c3, y_train_c3))
]

means_list = [
    [np.mean(mean_x_c1), np.mean(mean_y_c1)],
    [np.mean(mean_x_c1), np.mean(mean_y_c1)],
    [np.mean(mean_x_c1), np.mean(mean_y_c1)]
]


# Initialize a list to store covariance matrices for each case
covariance_matrices = []

# Calculate covariance matrices for different cases and classes
covariance_matrix_c1_case1 = covariance_matrix_c2_case1 = covariance_matrix_c3_case1 = calculate_covariance_matrices(data_list, means_list, case=1)
covariance_matrix_c1_case2 = covariance_matrix_c2_case2 = covariance_matrix_c3_case2 = calculate_covariance_matrices(data_list, means_list, case=2)
covariance_matrix_c1_case3, covariance_matrix_c2_case3, covariance_matrix_c3_case3 = calculate_covariance_matrices(data_list, means_list, case=3)
covariance_matrix_c1_case4, covariance_matrix_c2_case4, covariance_matrix_c3_case4 = calculate_covariance_matrices(data_list, means_list, case=4)
covariance_matrices.append([covariance_matrix_c1_case1 ,covariance_matrix_c2_case1 , covariance_matrix_c3_case1])
covariance_matrices.append([covariance_matrix_c1_case2 ,covariance_matrix_c2_case2 , covariance_matrix_c3_case2])
covariance_matrices.append([covariance_matrix_c1_case3 ,covariance_matrix_c2_case3 , covariance_matrix_c3_case3])
covariance_matrices.append([covariance_matrix_c1_case4 ,covariance_matrix_c2_case4 , covariance_matrix_c3_case4])

#confusion_data = list()
#for i, cov in enumerate(covariance_matrices):
# Create a single plot for all three classes with training data and mean
plt.figure(figsize=(10, 6))

# Class 1
plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')

# Class 2
plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')

# Class 3
plt.scatter(X_train_c3, y_train_c3, c='darkgreen', label='Class 3 Training Data')

# Class 1
plt.scatter(X_test_c1, y_test_c1, c='darkred', label='Class 1 Training Data')

# Class 2
plt.scatter(X_test_c2, y_test_c2, c='darkblue', label='Class 2 Training Data')

# Class 3
plt.scatter(X_test_c3, y_test_c3, c='darkgreen', label='Class 3 Training Data')

plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label, y_train_c3_label))

# Training logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter( combined_array_train_c1[:, 0],  combined_array_train_c1[:, 1], label='Class C1')
plt.scatter( combined_array_train_c2[:, 0],  combined_array_train_c2[:, 1], label='Class C2')
plt.scatter( combined_array_train_c3[:, 0],  combined_array_train_c3[:, 1], label='Class C3')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Logistic Regression for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label))

# Training logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter( combined_array_train_c1[:, 0],  combined_array_train_c1[:, 1], label='Class C1')
plt.scatter( combined_array_train_c2[:, 0],  combined_array_train_c2[:, 1], label='Class C2')
#plt.scatter( combined_array_train_c3[:, 0],  combined_array_train_c3[:, 1], label='Class C3')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Logistic Regression for LS dataset1')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate((combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))

# Training logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
#plt.scatter( combined_array_train_c1[:, 0],  combined_array_train_c1[:, 1], label='Class C1')
plt.scatter( combined_array_train_c2[:, 0],  combined_array_train_c2[:, 1], label='Class C2')
plt.scatter( combined_array_train_c3[:, 0],  combined_array_train_c3[:, 1], label='Class C3')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Logistic Regression for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c3))
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))

# Training logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter( combined_array_train_c1[:, 0],  combined_array_train_c1[:, 1], label='Class C1')
#plt.scatter( combined_array_train_c2[:, 0],  combined_array_train_c2[:, 1], label='Class C2')
plt.scatter( combined_array_train_c3[:, 0],  combined_array_train_c3[:, 1], label='Class C3')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Logistic Regression for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=1) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
#plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=1) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c3))
y_train = np.concatenate((y_train_c1_label , y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
#plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=1) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate((  combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate(( y_train_c2_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
#plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=1) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=2) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
#plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=2) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c3))
y_train = np.concatenate((y_train_c1_label , y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
#plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=2) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate((  combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate(( y_train_c2_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
#plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=2) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label, y_train_c3_label))

# Creating and fitting the LDA model
lda = LDA()
lda.fit(X_train, y_train)

# Creating a meshgrid to plot decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predicting for each point on the meshgrid
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Linear Discriminant Analysis (LDA) for LS dataset1')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# Concatenating data for training
X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label))

# Concatenate the data for FLDA
X_train = np.concatenate((np.concatenate((combined_array_train_c1, combined_array_train_c2)), combined_array_train_c3))
y_train = np.concatenate((np.concatenate((y_train_c1_label, y_train_c2_label)), y_train_c3_label))

# Fit FLDA model
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)


# Create a meshgrid to cover the feature space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Use FLDA to predict classes for meshgrid points
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Linear Discriminant Analysis (LDA) for LS dataset1')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2



# Concatenate the data for FLDA
X_train = np.concatenate((np.concatenate((combined_array_train_c1, combined_array_train_c2)), combined_array_train_c3))
y_train = np.concatenate((np.concatenate((y_train_c1_label, y_train_c2_label)), y_train_c3_label))

# Fit FLDA model
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train, y_train)


# Create a meshgrid to cover the feature space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Use FLDA to predict classes for meshgrid points
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Linear Discriminant Analysis (LDA) for LS dataset1')
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# # Concatenating data for training
# X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2))
# y_train = np.concatenate((y_train_c1_label, y_train_c2_label))

# Concatenate the data for FLDA
X_train = np.concatenate((combined_array_train_c1, combined_array_train_c2))
y_train = np.concatenate((y_train_c1_label, y_train_c2_label))

# Fit FLDA model
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train, y_train)


# Create a meshgrid to cover the feature space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Use FLDA to predict classes for meshgrid points
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
#plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Linear Discriminant Analysis (LDA) for LS dataset1')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# # Concatenating data for training
# X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2))
# y_train = np.concatenate((y_train_c1_label, y_train_c2_label))

# Concatenate the data for FLDA
X_train = np.concatenate((combined_array_train_c2, combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))

# Fit FLDA model
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train, y_train)


# Create a meshgrid to cover the feature space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Use FLDA to predict classes for meshgrid points
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
#plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Linear Discriminant Analysis (LDA) for LS dataset1')
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Creating data for each class (replace this with your actual data)
#X_train_c1 = np.array([[1, 2], [2, 3], [3, 3], [2, 1]])
#y_train_c1 = np.array([0, 0, 0, 0])  # Assuming class 1 is labeled as 0

# Creating a 2D array using a for loop
combined_array_train_c1 = np.zeros((len(X_train_c1), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c1)):
    combined_array_train_c1[i][0] = X_train_c1[i]  # Assigning values from X_train_c1
    combined_array_train_c1[i][1] = y_train_c1[i]  # Assigning values from y_train_c1
    
# Example length of y_train_c1
length_y_train_c1 = len(y_train_c1) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c1_label = np.zeros(length_y_train_c1)  # Initialize a 1D array of zeros

for i in range(length_y_train_c1):
    y_train_c1_label[i] = 0  # Assigning value 1 to each element

    
    
    
# Creating a 2D array using a for loop
combined_array_train_c2= np.zeros((len(X_train_c2), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c2)):
    combined_array_train_c2[i][0] = X_train_c2[i]  # Assigning values from X_train_c2
    combined_array_train_c2[i][1] = y_train_c2[i]  # Assigning values from y_train_c2
    

# Example length of y_train_c1
length_y_train_c2 = len(y_train_c2) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c2_label = np.zeros(length_y_train_c2)  # Initialize a 1D array of ones

for i in range(length_y_train_c2):
    y_train_c2_label[i] = 1  # Assigning value 1 to each element
    
    
    

# Creating a 2D array using a for loop
combined_array_train_c3= np.zeros((len(X_train_c3), 2))  # Initialize a 2D array of zeros

for i in range(len(X_train_c3)):
    combined_array_train_c3[i][0] = X_train_c3[i]  # Assigning values from X_train_c3
    combined_array_train_c3[i][1] = y_train_c3[i]  # Assigning values from y_train_c3
    

# Example length of y_train_c1
length_y_train_c3 = len(y_train_c3) # Replace this with the actual length

# Creating a 1D array of all 1's using a for loop
y_train_c3_label = np.zeros(length_y_train_c3)  # Initialize a 1D array of zeroes

for i in range(length_y_train_c3):
    y_train_c3_label[i] = 2  # Assigning value 2 to each element
    

#print(ones_array)

#print(combined_array)

#X_train_c2 = np.array([[3, 5], [5, 1], [4, 6], [6, 3]])
#y_train_c2 = np.array([1, 1, 1, 1])  # Assuming class 2 is labeled as 1

#X_train_c3 = np.array([[7, 8], [8, 9], [9, 7], [8, 8]])
#y_train_c3 = np.array([2, 2, 2, 2])  # Assuming class 3 is labeled as 2

# # Concatenating data for training
# X_train = np.concatenate(( combined_array_train_c1,  combined_array_train_c2))
# y_train = np.concatenate((y_train_c1_label, y_train_c2_label))

# Concatenate the data for FLDA
X_train = np.concatenate((combined_array_train_c1, combined_array_train_c3))
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))

# Fit FLDA model
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train, y_train)


# Create a meshgrid to cover the feature space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Use FLDA to predict classes for meshgrid points
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
#plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with Linear Discriminant Analysis (LDA) for LS dataset1')
plt.legend()
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