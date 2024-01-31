# -*- coding: utf-8 -*-
"""pr_assignment1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M9Q0b0FNg5LEBDTil0xjp5vybLqZvLS-
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import math


def split_data(X, y):
    l = X.shape[0]

    # Calculate the train data size (70%)
    train_data_size = int(l * 0.7)

    # Initialize train and test data lists
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    # Populate the train and test data lists
    for i in range(train_data_size):
        X_train.append(X[i])
        y_train.append(y[i])
    for i in range(train_data_size, l):
        X_test.append(X[i])
        y_test.append(y[i])
    return X_train, X_test, y_train, y_test


def calculate_mean(arr):
    total = 0
    count = 0
    for element in arr:
        total += element
        count += 1
    if count == 0:
        return 0
    return total / count


# Custom transposition function
def custom_transpose(arr):
    return np.array(arr).T


# Custom covariance calculation function
def custom_covariance(data):
    num_samples, num_features = data.shape
    mean_vector = calculate_mean(data)
    centered_data = data - mean_vector
    covariance_matrix = np.zeros((num_features, num_features))
    for i in range(num_samples):
        for j in range(num_features):
            for k in range(num_features):
                covariance_matrix[j, k] += centered_data[i, j] * centered_data[i, k]

    covariance_matrix /= num_samples - 1
    return covariance_matrix


# Function to calculate covariance matrices for different cases
def calculate_covariance_matrices(data_list, case):
    x = data_list[0]
    y = data_list[1]
    z = data_list[2]

    if case == 1:
        A = custom_covariance(x)
        B = custom_covariance(y)
        C = custom_covariance(z)
        t = np.add(A, B)
        avg = (1 / 3) * np.add(t, C)
        var = np.diag(avg)
        avg_var = np.sum(var) / len(var)
        cov_mat = np.identity(len(avg)) * avg_var
        return cov_mat, cov_mat, cov_mat

    elif case == 2:
        A = custom_covariance(x)
        B = custom_covariance(y)
        C = custom_covariance(z)
        t = np.add(A, B)
        avg_cov_mat = (1 / 3) * np.add(t, C)

        return avg_cov_mat, avg_cov_mat, avg_cov_mat

    elif case == 3:
        A = custom_covariance(x)
        B = custom_covariance(y)
        C = custom_covariance(z)
        a = np.diag(A)
        b = np.diag(B)
        c = np.diag(C)
        cov_mat_x = np.identity(len(a)) * custom_transpose(a)
        cov_mat_y = np.identity(len(b)) * custom_transpose(b)
        cov_mat_z = np.identity(len(c)) * custom_transpose(c)

        return cov_mat_x, cov_mat_y, cov_mat_z

    elif case == 4:
        A = custom_covariance(x)
        B = custom_covariance(y)
        C = custom_covariance(z)
        return A, B, C

    else:
        raise ValueError("Invalid case. Choose from 1, 2, 3, or 4.")


def calculate_log_likelihood_single(data_point, mean, cov_matrix, prior):
    centered_x = np.array(data_point) - np.array(mean)
    inv_cov_mat = np.linalg.inv(cov_matrix)
    det_cov_mat = np.linalg.det(cov_matrix)
    log_likelihood = (
        (-1 / 2)
        * np.matmul(np.matmul(np.transpose(centered_x), inv_cov_mat), centered_x)
        - (1 / 2) * math.log(det_cov_mat)
        + math.log(prior)
    )
    return log_likelihood


file_path_c1 = "/content/Class1.txt"
data_c1 = np.genfromtxt(file_path_c1, delimiter=" ")
X_c1, y_c1 = data_c1[:, 0], data_c1[:, 1]

X_train_c1, X_test_c1, y_train_c1, y_test_c1 = split_data(X_c1, y_c1)

mean_x_c1 = calculate_mean(X_train_c1)
mean_y_c1 = calculate_mean(y_train_c1)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_c1, y_train_c1, c="darkred", label="Training Data")
plt.scatter(mean_x_c1, mean_y_c1, c="black", label="Mean")
plt.title("Class 1 Training Data")
plt.legend()
plt.show()

file_path_c2 = "/content/Class2.txt"
data_c2 = np.genfromtxt(file_path_c2, delimiter=" ")
X_c2, y_c2 = data_c2[:, 0], data_c2[:, 1]
X_train_c2, X_test_c2, y_train_c2, y_test_c2 = split_data(X_c2, y_c2)

mean_x_c2 = calculate_mean(X_train_c2)
mean_y_c2 = calculate_mean(y_train_c2)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_c2, y_train_c2, c="darkblue", label="Training Data")
plt.scatter(mean_x_c2, mean_y_c2, c="black", label="Mean")
plt.title("Class 2 Training Data")
plt.legend()
plt.show()

file_path_c3 = "/content/Class3.txt"
data_c3 = np.genfromtxt(file_path_c3, delimiter=" ")
X_c3, y_c3 = data_c3[:, 0], data_c3[:, 1]
X_train_c3, X_test_c3, y_train_c3, y_test_c3 = split_data(X_c3, y_c3)

mean_x_c3 = calculate_mean(X_train_c3)
mean_y_c3 = calculate_mean(y_train_c3)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_c3, y_train_c3, c="darkgreen", label="Training Data")
plt.scatter(mean_x_c3, mean_y_c3, c="black", label="Mean")
plt.title("Class 3 Training Data")
plt.legend()
plt.show()

lenC1 = len(X_c1)
lenC2 = len(X_c2)
lenC3 = len(X_c3)
total = lenC1 + lenC2 + lenC3
prior1 = float(lenC1 / total)
prior2 = float(lenC2 / total)
prior3 = float(lenC3 / total)

plt.figure(figsize=(10, 6))

plt.scatter(X_train_c1, y_train_c1, c="darkred", label="Class 1 Training Data")
plt.scatter(mean_x_c1, mean_y_c1, c="black", label="Class 1 Mean")

plt.scatter(X_train_c2, y_train_c2, c="darkblue", label="Class 2 Training Data")
plt.scatter(mean_x_c2, mean_y_c2, c="black", label="Class 2 Mean")

plt.scatter(X_train_c3, y_train_c3, c="darkgreen", label="Class 3 Training Data")
plt.scatter(mean_x_c3, mean_y_c3, c="black", label="Class 3 Mean")

plt.title("Training Data and Mean for Three Classes")
plt.legend()
plt.show()

# Data and means lists for different cases
data_list = [
    np.column_stack((X_train_c1, y_train_c1)),
    np.column_stack((X_train_c2, y_train_c2)),
    np.column_stack((X_train_c3, y_train_c3)),
]

# Calculate covariance matrices for different classes
covariance_matrices_classes = []
for data in data_list:
    cov_matrices = custom_covariance(data)
    covariance_matrices_classes.append(cov_matrices)

# Calculate covariance matrices for different classes
covariance_matrices_cases = []
for i in range(1, 5):
    cov_matrices = calculate_covariance_matrices(data_list, i)
    covariance_matrices_cases.append(cov_matrices)

confusion_data = list()
for i, cov in enumerate(covariance_matrices_cases):
    # Create a single plot for all three classes with training data and mean
    plt.figure(figsize=(10, 6))

    # Class 1
    plt.scatter(X_train_c1, y_train_c1, c="darkred", label="Class 1 Training Data")

    # Class 2
    plt.scatter(X_train_c2, y_train_c2, c="darkblue", label="Class 2 Training Data")

    # Class 3
    plt.scatter(X_train_c3, y_train_c3, c="darkgreen", label="Class 3 Training Data")

    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(X_test_c1)):
        # Calculate log-likelihood for Class 1
        c1 = calculate_log_likelihood_single(
            [X_test_c1[i], y_test_c1[i]], [mean_x_c1, mean_y_c1], cov[0], prior1
        )

        # Calculate log-likelihood for Class 2
        c2 = calculate_log_likelihood_single(
            [X_test_c1[i], y_test_c1[i]], [mean_x_c2, mean_y_c2], cov[1], prior2
        )

        # Calculate log-likelihood for Class 3
        c3 = calculate_log_likelihood_single(
            [X_test_c1[i], y_test_c1[i]], [mean_x_c3, mean_y_c3], cov[2], prior3
        )
        maxx = max(c1, c2, c3)

        if i == 1:
            if maxx == c1:
                confusion[0][0] += 1
                plt.scatter(
                    X_test_c1[i], y_test_c1[i], c="magenta", label="Class 1 Test Data"
                )
            elif maxx == c2:
                confusion[1][0] += 1
                plt.scatter(
                    X_test_c1[i], y_test_c1[i], c="magenta", label="Class 2 Test Data"
                )
            else:
                confusion[2][0] += 1
                plt.scatter(
                    X_test_c1[i], y_test_c1[i], c="magenta", label="Class 3 Test Data"
                )
        else:
            if maxx == c1:
                confusion[0][0] += 1
                plt.scatter(X_test_c1[i], y_test_c1[i], c="magenta")
            elif maxx == c2:
                confusion[1][0] += 1
                plt.scatter(X_test_c1[i], y_test_c1[i], c="magenta")
            else:
                confusion[2][0] += 1
                plt.scatter(X_test_c1[i], y_test_c1[i], c="magenta")
    for i in range(len(X_test_c2)):
        # Calculate log-likelihood for Class 1
        c1 = calculate_log_likelihood_single(
            [X_test_c2[i], y_test_c2[i]], [mean_x_c1, mean_y_c1], cov[0], prior1
        )

        # Calculate log-likelihood for Class 2
        c2 = calculate_log_likelihood_single(
            [X_test_c2[i], y_test_c2[i]], [mean_x_c2, mean_y_c2], cov[1], prior2
        )

        # Calculate log-likelihood for Class 3
        c3 = calculate_log_likelihood_single(
            [X_test_c2[i], y_test_c2[i]], [mean_x_c3, mean_y_c3], cov[2], prior3
        )

        maxx = max(c1, c2, c3)

        if i == 1:
            if maxx == c1:
                confusion[0][1] += 1
                plt.scatter(
                    X_test_c2[i], y_test_c2[i], c="cyan", label="Class 1 Test Data"
                )
            elif maxx == c2:
                confusion[1][1] += 1
                plt.scatter(
                    X_test_c2[i], y_test_c2[i], c="cyan", label="Class 2 Test Data"
                )
            else:
                confusion[2][1] += 1
                plt.scatter(
                    X_test_c2[i], y_test_c2[i], c="cyan", label="Class 3 Test Data"
                )
        else:
            if maxx == c1:
                confusion[0][1] += 1
                plt.scatter(X_test_c2[i], y_test_c2[i], c="cyan")
            elif maxx == c2:
                confusion[1][1] += 1
                plt.scatter(X_test_c2[i], y_test_c2[i], c="cyan")
            else:
                confusion[2][1] += 1
                plt.scatter(X_test_c2[i], y_test_c2[i], c="cyan")

    for i in range(len(X_test_c3)):
        # Calculate log-likelihood for Class 1
        c1 = calculate_log_likelihood_single(
            [X_test_c3[i], y_test_c3[i]], [mean_x_c1, mean_y_c1], cov[0], prior1
        )

        # Calculate log-likelihood for Class 2
        c2 = calculate_log_likelihood_single(
            [X_test_c3[i], y_test_c3[i]], [mean_x_c2, mean_y_c2], cov[1], prior2
        )

        # Calculate log-likelihood for Class 3
        c3 = calculate_log_likelihood_single(
            [X_test_c3[i], y_test_c3[i]], [mean_x_c3, mean_y_c3], cov[2], prior3
        )

        maxx = max(c1, c2, c3)

        if i == 1:
            if maxx == c1:
                confusion[0][2] += 1
                plt.scatter(
                    X_test_c3[i], y_test_c3[i], c="yellow", label="Class 1 Test Data"
                )
            elif maxx == c2:
                confusion[1][2] += 1
                plt.scatter(
                    X_test_c3[i], y_test_c3[i], c="yellow", label="Class 2 Test Data"
                )
            else:
                confusion[2][2] += 1
                plt.scatter(
                    X_test_c3[i], y_test_c3[i], c="yellow", label="Class 3 Test Data"
                )
        else:
            if maxx == c1:
                confusion[0][2] += 1
                plt.scatter(X_test_c3[i], y_test_c3[i], c="yellow")
            elif maxx == c2:
                confusion[1][2] += 1
                plt.scatter(X_test_c3[i], y_test_c3[i], c="yellow")
            else:
                confusion[2][2] += 1
                plt.scatter(X_test_c3[i], y_test_c3[i], c="yellow")

    plt.scatter(mean_x_c1, mean_y_c1, c="black", label="Class 1 Mean")
    plt.scatter(mean_x_c2, mean_y_c2, c="black", label="Class 2 Mean")
    plt.scatter(mean_x_c3, mean_y_c3, c="black", label="Class 3 Mean")
    plt.title("Training Data and Mean for Three Classes")
    plt.legend()
    plt.show()
    confusion_data.append(confusion)

    # Find the minimum and maximum values for X and Y from your training data
    min_x = min(np.min(X_train_c1), np.min(X_train_c2), np.min(X_train_c3))
    max_x = max(np.max(X_train_c1), np.max(X_train_c2), np.max(X_train_c3))
    min_y = min(np.min(y_train_c1), np.min(y_train_c2), np.min(y_train_c3))
    max_y = max(np.max(y_train_c1), np.max(y_train_c2), np.max(y_train_c3))

    # Define the number of points and create a denser meshgrid
    num_points = 50  # Increase this value for a denser mesh
    X_color = np.linspace(min_x - 5, max_x + 5, num_points)
    Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

    # Create a meshgrid of points in the X-Y plane
    X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)

    # Create an empty array to store class labels for each point in the mesh
    class_labels = np.zeros(X_mesh.shape)

    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            x_val = X_mesh[i, j]
            y_val = Y_mesh[i, j]

            G_color_1 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c1, mean_y_c1], cov[0], prior1
            )
            G_color_2 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c2, mean_y_c2], cov[1], prior2
            )
            G_color_3 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c3, mean_y_c3], cov[2], prior3
            )

            # Find the class with the maximum likelihood
            max_likelihood_class = np.argmax([G_color_1, G_color_2, G_color_3])

            # Store the class label in the corresponding location in the array
            class_labels[i, j] = max_likelihood_class

    # Now, you can plot the denser mesh with colors based on class labels
    plt.scatter(X_train_c1, y_train_c1, c="darkred", label="Class 1 Training Data")
    plt.scatter(X_train_c2, y_train_c2, c="darkblue", label="Class 2 Training Data")
    plt.scatter(X_train_c3, y_train_c3, c="darkgreen", label="Class 3 Training Data")
    plt.contourf(
        X_mesh,
        Y_mesh,
        class_labels,
        levels=[0, 0.5, 1.5, 2.5],
        colors=["r", "g", "y"],
        alpha=0.3,
    )
    plt.legend()
    plt.show()

for i, cov in enumerate(covariance_matrices):
    # Create a single plot for all three classes with training data and mean
    plt.figure(figsize=(10, 6))

    # Class 1
    plt.scatter(X_train_c1, y_train_c1, c="darkred", label="Class 1 Training Data")

    # Class 2
    plt.scatter(X_train_c2, y_train_c2, c="darkblue", label="Class 2 Training Data")

    for i in range(len(X_test_c1)):
        # Calculate log-likelihood for Class 1
        c1 = calculate_log_likelihood_single(
            [X_test_c1[i], y_test_c1[i]], [mean_x_c1, mean_y_c1], cov[0], prior1
        )

        # Calculate log-likelihood for Class 2
        c2 = calculate_log_likelihood_single(
            [X_test_c1[i], y_test_c1[i]], [mean_x_c2, mean_y_c2], cov[1], prior2
        )

        maxx = max(c1, c2)

        if i == 1:
            if maxx == c1:
                plt.scatter(
                    X_test_c1[i], y_test_c1[i], c="magenta", label="Class 1 Test Data"
                )
            else:
                plt.scatter(
                    X_test_c1[i], y_test_c1[i], c="magenta", label="Class 2 Test Data"
                )
        else:
            if maxx == c1:
                plt.scatter(X_test_c1[i], y_test_c1[i], c="magenta")
            else:
                plt.scatter(X_test_c1[i], y_test_c1[i], c="magenta")

    for i in range(len(X_test_c2)):
        # Calculate log-likelihood for Class 1
        c1 = calculate_log_likelihood_single(
            [X_test_c2[i], y_test_c2[i]], [mean_x_c1, mean_y_c1], cov[0], prior1
        )

        # Calculate log-likelihood for Class 2
        c2 = calculate_log_likelihood_single(
            [X_test_c2[i], y_test_c2[i]], [mean_x_c2, mean_y_c2], cov[1], prior2
        )

        maxx = max(c1, c2)

        if i == 1:
            if maxx == c1:
                plt.scatter(
                    X_test_c2[i], y_test_c2[i], c="cyan", label="Class 1 Test Data"
                )
            else:
                plt.scatter(
                    X_test_c2[i], y_test_c2[i], c="cyan", label="Class 2 Test Data"
                )
        else:
            if maxx == c1:
                plt.scatter(X_test_c2[i], y_test_c2[i], c="cyan")
            else:
                plt.scatter(X_test_c2[i], y_test_c2[i], c="cyan")

    plt.scatter(mean_x_c1, mean_y_c1, c="black", label="Class 1 Mean")
    plt.scatter(mean_x_c2, mean_y_c2, c="black", label="Class 2 Mean")
    plt.title("Training Data and Mean for Two Classes")
    plt.legend()
    plt.show()

    # Find the minimum and maximum values for X and Y from your training data
    min_x = min(np.min(X_train_c1), np.min(X_train_c2))
    max_x = max(np.max(X_train_c1), np.max(X_train_c2))
    min_y = min(np.min(y_train_c1), np.min(y_train_c2))
    max_y = max(np.max(y_train_c1), np.max(y_train_c2))

    # Define the number of points and create a denser meshgrid
    num_points = 50  # Increase this value for a denser mesh
    X_color = np.linspace(min_x - 5, max_x + 5, num_points)
    Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

    # Create a meshgrid of points in the X-Y plane
    X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)

    # Create an empty array to store class labels for each point in the mesh
    class_labels = np.zeros(X_mesh.shape)

    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            x_val = X_mesh[i, j]
            y_val = Y_mesh[i, j]

            G_color_1 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c1, mean_y_c1], cov[0], prior1
            )
            G_color_2 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c2, mean_y_c2], cov[1], prior2
            )

            # Find the class with the maximum likelihood
            max_likelihood_class = np.argmax([G_color_1, G_color_2])

            # Store the class label in the corresponding location in the array
            class_labels[i, j] = max_likelihood_class

    # Now, you can plot the denser mesh with colors based on class labels
    plt.scatter(X_train_c1, y_train_c1, c="darkred", label="Class 1 Training Data")
    plt.scatter(X_train_c2, y_train_c2, c="darkblue", label="Class 2 Training Data")
    plt.contourf(
        X_mesh,
        Y_mesh,
        class_labels,
        levels=[0, 0.5, 1.5, 2.5],
        colors=["r", "g"],
        alpha=0.3,
    )
    plt.legend()
    plt.show()

    ####################################

    # Create a single plot for all three classes with training data and mean
    plt.figure(figsize=(10, 6))

    # Class 2
    plt.scatter(X_train_c2, y_train_c2, c="darkblue", label="Class 2 Training Data")

    # Class 3
    plt.scatter(X_train_c3, y_train_c3, c="darkgreen", label="Class 3 Training Data")

    for i in range(len(X_test_c2)):
        # Calculate log-likelihood for Class 2
        c2 = calculate_log_likelihood_single(
            [X_test_c2[i], y_test_c2[i]], [mean_x_c2, mean_y_c2], cov[1], prior2
        )

        # Calculate log-likelihood for Class 3
        c3 = calculate_log_likelihood_single(
            [X_test_c2[i], y_test_c2[i]], [mean_x_c3, mean_y_c3], cov[2], prior3
        )

        maxx = max(c2, c3)

        if i == 1:
            if maxx == c2:
                plt.scatter(
                    X_test_c2[i], y_test_c2[i], c="cyan", label="Class 2 Test Data"
                )
            else:
                plt.scatter(
                    X_test_c2[i], y_test_c2[i], c="cyan", label="Class 3 Test Data"
                )
        else:
            if maxx == c2:
                plt.scatter(X_test_c2[i], y_test_c2[i], c="cyan")
            else:
                plt.scatter(X_test_c2[i], y_test_c2[i], c="cyan")

    for i in range(len(X_test_c3)):
        # Calculate log-likelihood for Class 2
        c2 = calculate_log_likelihood_single(
            [X_test_c3[i], y_test_c3[i]], [mean_x_c2, mean_y_c2], cov[1], prior2
        )

        # Calculate log-likelihood for Class 3
        c3 = calculate_log_likelihood_single(
            [X_test_c3[i], y_test_c3[i]], [mean_x_c3, mean_y_c3], cov[2], prior3
        )

        maxx = max(c2, c3)

        if i == 1:
            if maxx == c2:
                plt.scatter(
                    X_test_c3[i], y_test_c3[i], c="yellow", label="Class 2 Test Data"
                )
            else:
                plt.scatter(
                    X_test_c3[i], y_test_c3[i], c="yellow", label="Class 3 Test Data"
                )
        else:
            if maxx == c2:
                plt.scatter(X_test_c3[i], y_test_c3[i], c="yellow")
            else:
                plt.scatter(X_test_c3[i], y_test_c3[i], c="yellow")

    plt.scatter(mean_x_c2, mean_y_c2, c="black", label="Class 2 Mean")
    plt.scatter(mean_x_c3, mean_y_c3, c="black", label="Class 3 Mean")
    plt.title("Training Data and Mean for Two Classes")
    plt.legend()
    plt.show()

    # Find the minimum and maximum values for X and Y from your training data
    min_x = min(np.min(X_train_c2), np.min(X_train_c3))
    max_x = max(np.max(X_train_c2), np.max(X_train_c3))
    min_y = min(np.min(y_train_c2), np.min(y_train_c3))
    max_y = max(np.max(y_train_c2), np.max(y_train_c3))

    # Define the number of points and create a denser meshgrid
    num_points = 50  # Increase this value for a denser mesh
    X_color = np.linspace(min_x - 5, max_x + 5, num_points)
    Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

    # Create a meshgrid of points in the X-Y plane
    X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)

    # Create an empty array to store class labels for each point in the mesh
    class_labels = np.zeros(X_mesh.shape)

    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            x_val = X_mesh[i, j]
            y_val = Y_mesh[i, j]

            G_color_1 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c2, mean_y_c2], cov[1], prior2
            )
            G_color_2 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c3, mean_y_c3], cov[2], prior3
            )

            # Find the class with the maximum likelihood
            max_likelihood_class = np.argmax([G_color_1, G_color_2])

            # Store the class label in the corresponding location in the array
            class_labels[i, j] = max_likelihood_class

    # Now, you can plot the denser mesh with colors based on class labels
    plt.scatter(X_train_c2, y_train_c2, c="darkblue", label="Class 2 Training Data")
    plt.scatter(X_train_c3, y_train_c3, c="darkgreen", label="Class 3 Training Data")
    plt.contourf(
        X_mesh,
        Y_mesh,
        class_labels,
        levels=[0, 0.5, 1.5, 2.5],
        colors=["g", "y"],
        alpha=0.3,
    )
    plt.legend()
    plt.show()

    ####################################

    # Create a single plot for all three classes with training data and mean
    plt.figure(figsize=(10, 6))

    # Class 1
    plt.scatter(X_train_c1, y_train_c1, c="darkred", label="Class 1 Training Data")

    # Class 3
    plt.scatter(X_train_c3, y_train_c3, c="darkgreen", label="Class 3 Training Data")

    for i in range(len(X_test_c1)):
        # Calculate log-likelihood for Class 1
        c1 = calculate_log_likelihood_single(
            [X_test_c1[i], y_test_c1[i]], [mean_x_c1, mean_y_c1], cov[0], prior1
        )

        # Calculate log-likelihood for Class 3
        c3 = calculate_log_likelihood_single(
            [X_test_c1[i], y_test_c1[i]], [mean_x_c3, mean_y_c3], cov[2], prior3
        )

        maxx = max(c1, c3)

        if i == 1:
            if maxx == c1:
                plt.scatter(
                    X_test_c1[i], y_test_c1[i], c="magenta", label="Class 1 Test Data"
                )
            else:
                plt.scatter(
                    X_test_c1[i], y_test_c1[i], c="magenta", label="Class 3 Test Data"
                )
        else:
            if maxx == c1:
                plt.scatter(X_test_c1[i], y_test_c1[i], c="magenta")
            else:
                plt.scatter(X_test_c1[i], y_test_c1[i], c="magenta")

    for i in range(len(X_test_c3)):
        # Calculate log-likelihood for Class 1
        c1 = calculate_log_likelihood_single(
            [X_test_c3[i], y_test_c3[i]], [mean_x_c1, mean_y_c1], cov[0], prior1
        )

        # Calculate log-likelihood for Class 3
        c3 = calculate_log_likelihood_single(
            [X_test_c3[i], y_test_c3[i]], [mean_x_c3, mean_y_c3], cov[2], prior3
        )

        maxx = max(c1, c3)

        if i == 1:
            if maxx == c1:
                plt.scatter(
                    X_test_c3[i], y_test_c3[i], c="yellow", label="Class 1 Test Data"
                )
            else:
                plt.scatter(
                    X_test_c3[i], y_test_c3[i], c="yellow", label="Class 3 Test Data"
                )
        else:
            if maxx == c1:
                plt.scatter(X_test_c3[i], y_test_c3[i], c="yellow")
            else:
                plt.scatter(X_test_c3[i], y_test_c3[i], c="yellow")

    plt.scatter(mean_x_c1, mean_y_c1, c="black", label="Class 1 Mean")
    plt.scatter(mean_x_c3, mean_y_c3, c="black", label="Class 3 Mean")
    plt.title("Training Data and Mean for Two Classes")
    plt.legend()
    plt.show()

    # Find the minimum and maximum values for X and Y from your training data
    min_x = min(np.min(X_train_c1), np.min(X_train_c3))
    max_x = max(np.max(X_train_c1), np.max(X_train_c3))
    min_y = min(np.min(y_train_c1), np.min(y_train_c3))
    max_y = max(np.max(y_train_c1), np.max(y_train_c3))

    # Define the number of points and create a denser meshgrid
    num_points = 50  # Increase this value for a denser mesh
    X_color = np.linspace(min_x - 5, max_x + 5, num_points)
    Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

    # Create a meshgrid of points in the X-Y plane
    X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)

    # Create an empty array to store class labels for each point in the mesh
    class_labels = np.zeros(X_mesh.shape)

    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            x_val = X_mesh[i, j]
            y_val = Y_mesh[i, j]

            G_color_1 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c1, mean_y_c1], cov[0], prior1
            )
            G_color_2 = calculate_log_likelihood_single(
                [x_val, y_val], [mean_x_c3, mean_y_c3], cov[2], prior3
            )

            # Find the class with the maximum likelihood
            max_likelihood_class = np.argmax([G_color_1, G_color_2])

            # Store the class label in the corresponding location in the array
            class_labels[i, j] = max_likelihood_class

    # Now, you can plot the denser mesh with colors based on class labels
    plt.scatter(X_train_c1, y_train_c1, c="darkred", label="Class 1 Training Data")
    plt.scatter(X_train_c3, y_train_c3, c="darkgreen", label="Class 3 Training Data")
    plt.contourf(
        X_mesh,
        Y_mesh,
        class_labels,
        levels=[0, 0.5, 1.5, 2.5],
        colors=["r", "y"],
        alpha=0.3,
    )
    plt.legend()
    plt.show()

print("Covariance Matrix for class 1")
print(covariance_matrices_classes[0])
print()
print("Covariance Matrix for class 2")
print(covariance_matrices_classes[1])
print()
print("Covariance Matrix for class 3")
print(covariance_matrices_classes[2])
print()
print("Covariance matrix for case 1: ")
print(covariance_matrices_cases[0][0])
print()
print("Covariance matrix for case 2: ")
print(covariance_matrices_cases[1][0])
print()
print("Covariance matrix for case 3 of class 1: ")
print(covariance_matrices_cases[2][0])
print()
print("Covariance matrix for case 3 of class 2: ")
print(covariance_matrices_cases[2][1])
print()
print("Covariance matrix for case 3 of class 3: ")
print(covariance_matrices_cases[2][2])
print()
print("Covariance matrix for case 4 of class 1: ")
print(covariance_matrices_cases[3][0])
print()
print("Covariance matrix for case 4 of class 2: ")
print(covariance_matrices_cases[3][1])
print()
print("Covariance matrix for case 4 of class 3: ")
print(covariance_matrices_cases[3][2])
print()
# For case 1
print("Confusion matrix for CASE 1:")
cm = confusion_data[0]
for data in confusion_data[0]:
    print(data)
print()
# Class 1
print("Class 1:")
tp = cm[0][0]
tn = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
fp = cm[0][1] + cm[0][2]
fn = cm[1][0] + cm[2][0]
acc11 = (tp + tn) / (tp + tn + fp + fn)

pr11 = (tp) / (tp + fp)

rec11 = tp / (tp + fn)

f11 = (2 * pr11 * rec11) / (pr11 + rec11)

print(acc11)
print(pr11)
print(rec11)
print(f11)

# Class 2
print("Class 2:")
tp = cm[1][1]
tn = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
fp = cm[1][0] + cm[1][2]
fn = cm[0][1] + cm[2][1]
acc21 = (tp + tn) / (tp + tn + fp + fn)

pr21 = (tp) / (tp + fp)

rec21 = tp / (tp + fn)

f21 = (2 * pr21 * rec21) / (pr21 + rec21)

print(acc21)
print(pr21)
print(rec21)
print(f21)

# Class 3
print("Class 3:")
tp = cm[2][2]
tn = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
fp = cm[2][0] + cm[2][1]
fn = cm[0][2] + cm[1][2]
acc31 = (tp + tn) / (tp + tn + fp + fn)

pr31 = (tp) / (tp + fp)

rec31 = tp / (tp + fn)

f31 = (2 * pr31 * rec31) / (pr31 + rec31)

print(acc31)
print(pr31)
print(rec31)
print(f31)

# Mean of Accuracy, precision, recall and F1 score.
avg_acc = (acc11 + acc21 + acc31) / 3
avg_pr = (pr11 + pr21 + pr31) / 3
avg_rec = (rec11 + rec21 + rec31) / 3
avg_f = (f11 + f21 + f31) / 3
print("Avg accuracy:", avg_acc)
print("Avg precision:", avg_pr)
print("Avg recall:", avg_rec)
print("Avg f1:", avg_f)

# For case 2
print("Confusion matrix for CASE 2:")
cm = confusion_data[1]
for data in confusion_data[1]:
    print(data)
print()
# Class 1
print("Class 1:")
tp = cm[0][0]
tn = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
fp = cm[0][1] + cm[0][2]
fn = cm[1][0] + cm[2][0]
acc11 = (tp + tn) / (tp + tn + fp + fn)

pr11 = (tp) / (tp + fp)

rec11 = tp / (tp + fn)

f11 = (2 * pr11 * rec11) / (pr11 + rec11)

print(acc11)
print(pr11)
print(rec11)
print(f11)

# Class 2
print("Class 2:")
tp = cm[1][1]
tn = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
fp = cm[1][0] + cm[1][2]
fn = cm[0][1] + cm[2][1]
acc21 = (tp + tn) / (tp + tn + fp + fn)

pr21 = (tp) / (tp + fp)

rec21 = tp / (tp + fn)

f21 = (2 * pr21 * rec21) / (pr21 + rec21)

print(acc21)
print(pr21)
print(rec21)
print(f21)

# Class 3
print("Class 3:")
tp = cm[2][2]
tn = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
fp = cm[2][0] + cm[2][1]
fn = cm[0][2] + cm[1][2]
acc31 = (tp + tn) / (tp + tn + fp + fn)

pr31 = (tp) / (tp + fp)

rec31 = tp / (tp + fn)

f31 = (2 * pr31 * rec31) / (pr31 + rec31)

print(acc31)
print(pr31)
print(rec31)
print(f31)

# Mean of Accuracy, precision, recall and F1 score.
avg_acc = (acc11 + acc21 + acc31) / 3
avg_pr = (pr11 + pr21 + pr31) / 3
avg_rec = (rec11 + rec21 + rec31) / 3
avg_f = (f11 + f21 + f31) / 3
print("Avg accuracy:", avg_acc)
print("Avg precision:", avg_pr)
print("Avg recall:", avg_rec)
print("Avg f1:", avg_f)

# For case 3
print("Confusion matrix for CASE 3:")
cm = confusion_data[2]
for data in confusion_data[2]:
    print(data)
print()
# Class 1
print("Class 1:")
tp = cm[0][0]
tn = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
fp = cm[0][1] + cm[0][2]
fn = cm[1][0] + cm[2][0]
acc11 = (tp + tn) / (tp + tn + fp + fn)

pr11 = (tp) / (tp + fp)

rec11 = tp / (tp + fn)

f11 = (2 * pr11 * rec11) / (pr11 + rec11)

print(acc11)
print(pr11)
print(rec11)
print(f11)

# Class 2
print("Class 2:")
tp = cm[1][1]
tn = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
fp = cm[1][0] + cm[1][2]
fn = cm[0][1] + cm[2][1]
acc21 = (tp + tn) / (tp + tn + fp + fn)

pr21 = (tp) / (tp + fp)

rec21 = tp / (tp + fn)

f21 = (2 * pr21 * rec21) / (pr21 + rec21)

print(acc21)
print(pr21)
print(rec21)
print(f21)

# Class 3
print("Class 3:")
tp = cm[2][2]
tn = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
fp = cm[2][0] + cm[2][1]
fn = cm[0][2] + cm[1][2]
acc31 = (tp + tn) / (tp + tn + fp + fn)

pr31 = (tp) / (tp + fp)

rec31 = tp / (tp + fn)

f31 = (2 * pr31 * rec31) / (pr31 + rec31)

print(acc31)
print(pr31)
print(rec31)
print(f31)

# Mean of Accuracy, precision, recall and F1 score.
avg_acc = (acc11 + acc21 + acc31) / 3
avg_pr = (pr11 + pr21 + pr31) / 3
avg_rec = (rec11 + rec21 + rec31) / 3
avg_f = (f11 + f21 + f31) / 3
print("Avg accuracy:", avg_acc)
print("Avg precision:", avg_pr)
print("Avg recall:", avg_rec)
print("Avg f1:", avg_f)

# For case 4
print("Confusion matrix for CASE 4:")
cm = confusion_data[3]
for data in confusion_data[3]:
    print(data)
print()
# Class 1
print("Class 1:")
tp = cm[0][0]
tn = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
fp = cm[0][1] + cm[0][2]
fn = cm[1][0] + cm[2][0]
acc11 = (tp + tn) / (tp + tn + fp + fn)

pr11 = (tp) / (tp + fp)

rec11 = tp / (tp + fn)

f11 = (2 * pr11 * rec11) / (pr11 + rec11)

print(acc11)
print(pr11)
print(rec11)
print(f11)

# Class 2
print("Class 2:")
tp = cm[1][1]
tn = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
fp = cm[1][0] + cm[1][2]
fn = cm[0][1] + cm[2][1]
acc21 = (tp + tn) / (tp + tn + fp + fn)

pr21 = (tp) / (tp + fp)

rec21 = tp / (tp + fn)

f21 = (2 * pr21 * rec21) / (pr21 + rec21)

print(acc21)
print(pr21)
print(rec21)
print(f21)

# Class 3
print("Class 3:")
tp = cm[2][2]
tn = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
fp = cm[2][0] + cm[2][1]
fn = cm[0][2] + cm[1][2]
acc31 = (tp + tn) / (tp + tn + fp + fn)

pr31 = (tp) / (tp + fp)

rec31 = tp / (tp + fn)

f31 = (2 * pr31 * rec31) / (pr31 + rec31)

print(acc31)
print(pr31)
print(rec31)
print(f31)

# Mean of Accuracy, precision, recall and F1 score.
avg_acc = (acc11 + acc21 + acc31) / 3
avg_pr = (pr11 + pr21 + pr31) / 3
avg_rec = (rec11 + rec21 + rec31) / 3
avg_f = (f11 + f21 + f31) / 3
print("Avg accuracy:", avg_acc)
print("Avg precision:", avg_pr)
print("Avg recall:", avg_rec)
print("Avg f1:", avg_f)