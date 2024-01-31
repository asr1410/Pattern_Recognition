import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import math


# In[2]:


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


# In[3]:


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


# In[4]:


file_path_c1 = "./data/NLS_1.txt"
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


# In[5]:


file_path_c2 = "./data/NLS_1.txt"
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


# In[6]:


file_path_c3 = "./data/NLS_1.txt"
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


# In[7]:


lenC1 = len(X_c1)
lenC2 = len(X_c2)
lenC3 = len(X_c3)
total =lenC1 + lenC2 + lenC3
print(total)
prior1 = float(lenC1/total)
prior2 = float(lenC2/total)
prior3 = float(lenC3/total)

plt.figure(figsize=(10, 6))


# In[8]:


plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')
plt.scatter(mean_x_c1, mean_y_c1, c='black', label='Class 1 Mean')


# In[9]:


plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')
plt.scatter(mean_x_c2, mean_y_c2, c='black', label='Class 2 Mean')


# In[10]:


plt.scatter(X_train_c3, y_train_c3, c='darkgreen', label='Class 3 Training Data')
plt.scatter(mean_x_c3, mean_y_c3, c='black', label='Class 3 Mean')


# In[11]:


plt.title('Training Data and Mean for Three Classes')
plt.legend()
plt.show()


# In[12]:



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


# In[13]:



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


# In[14]:


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


# In[15]:


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
plt.scatter(X_test_c1, y_test_c1, c='pink', label='Class 1 Training Data')

# Class 2
plt.scatter(X_test_c2, y_test_c2, c='magenta', label='Class 2 Training Data')

# Class 3
plt.scatter(X_test_c3, y_test_c3, c='yellow', label='Class 3 Training Data')

plt.legend()
plt.show()

   


# In[16]:


# confusion = [[0,0,0], [0,0,0], [0,0,0]]
# plt.figure(figsize=(10, 6))
# for i in range(len(X_test_c1)):
#         # Calculate log-likelihood for Class 1
#         c1 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c1, mean_y_c1], cov[0], prior1)

#         # Calculate log-likelihood for Class 2
#         c2 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c2, mean_y_c2], cov[1], prior2)

#         # Calculate log-likelihood for Class 3
#         c3 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c3, mean_y_c3], cov[2], prior3)
#         maxx = max(c1, c2, c3)

#         if i == 1:
#             if maxx == c1:
#                 confusion[0][0] += 1
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta', label="Class 1 Test Data")
#             elif maxx == c2:
#                 confusion[1][0] += 1
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta', label="Class 2 Test Data")
#             else:
#                 confusion[2][0] += 1
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta', label="Class 3 Test Data")
#         else:
#             if maxx == c1:
#                 confusion[0][0] += 1
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta')
#             elif maxx == c2:
#                 confusion[1][0] += 1
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta')
#             else:
#                 confusion[2][0] += 1
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta')
# for i in range(len(X_test_c2)):
#         # Calculate log-likelihood for Class 1
#         c1 = calculate_log_likelihood_single([X_test_c2[i], y_test_c2[i]], [mean_x_c1, mean_y_c1], cov[0], prior1)

#         # Calculate log-likelihood for Class 2
#         c2 = calculate_log_likelihood_single([X_test_c2[i], y_test_c2[i]], [mean_x_c2, mean_y_c2], cov[1], prior2)

#         # Calculate log-likelihood for Class 3
#         c3 = calculate_log_likelihood_single([X_test_c2[i], y_test_c2[i]], [mean_x_c3, mean_y_c3], cov[2], prior3)

#         maxx = max(c1, c2, c3)

#         if i == 1:
#             if maxx == c1:
#                 confusion[0][1] += 1
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan', label="Class 1 Test Data")
#             elif maxx == c2:
#                 confusion[1][1] += 1
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan', label="Class 2 Test Data")
#             else:
#                 confusion[2][1] += 1
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan', label="Class 3 Test Data")
#         else:
#             if maxx == c1:
#                 confusion[0][1] += 1
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan')
#             elif maxx == c2:
#                 confusion[1][1] += 1
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan')
#             else:
#                 confusion[2][1] += 1
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan')

# for i in range(len(X_test_c3)):
#         # Calculate log-likelihood for Class 1
#         c1 = calculate_log_likelihood_single([X_test_c3[i], y_test_c3[i]], [mean_x_c1, mean_y_c1], cov[0], prior1)

#         # Calculate log-likelihood for Class 2
#         c2 = calculate_log_likelihood_single([X_test_c3[i], y_test_c3[i]], [mean_x_c2, mean_y_c2], cov[1], prior2)

#         # Calculate log-likelihood for Class 3
#         c3 = calculate_log_likelihood_single([X_test_c3[i], y_test_c3[i]], [mean_x_c3, mean_y_c3], cov[2], prior3)

#         maxx = max(c1, c2, c3)

#         if i == 1:
#             if maxx == c1:
#                 confusion[0][2] += 1
#                 plt.scatter(X_test_c3[i], y_test_c3[i], c='yellow', label="Class 1 Test Data")
#             elif maxx == c2:
#                 confusion[1][2] += 1
#                 plt.scatter(X_test_c3[i], y_test_c3[i], c='yellow', label="Class 2 Test Data")
#             else:
#                 confusion[2][2] += 1
#                 plt.scatter(X_test_c3[i], y_test_c3[i], c='yellow', label="Class 3 Test Data")
#         else:
#             if maxx == c1:
#                 confusion[0][2] += 1
#                 plt.scatter(X_test_c3[i], y_test_c3[i], c='yellow')
#             elif maxx == c2:
#                 confusion[1][2] += 1
#                 plt.scatter(X_test_c3[i], y_test_c3[i], c='yellow')
#             else:
#                 confusion[2][2] += 1
#                 plt.scatter(X_test_c3[i], y_test_c3[i], c='yellow')


# plt.scatter(mean_x_c1, mean_y_c1, c='black', label='Class 1 Mean')
# plt.scatter(mean_x_c2, mean_y_c2, c='black', label='Class 2 Mean')
# plt.scatter(mean_x_c3, mean_y_c3, c='black', label='Class 3 Mean')
# plt.title('Training Data and Mean for Three Classes')
# # Class 1
# plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')

# # Class 2
# plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')

# # Class 3
# plt.scatter(X_train_c3, y_train_c3, c='darkgreen', label='Class 3 Training Data')

# plt.legend()
# plt.show()
# confusion_data.append(confusion)


# In[18]:


#     # Find the minimum and maximum values for X and Y from your training data
#     min_x = min(np.min(X_train_c1), np.min(X_train_c2), np.min(X_train_c3))
#     max_x = max(np.max(X_train_c1), np.max(X_train_c2), np.max(X_train_c3))
#     min_y = min(np.min(y_train_c1), np.min(y_train_c2), np.min(y_train_c3))
#     max_y = max(np.max(y_train_c1), np.max(y_train_c2), np.max(y_train_c3))

#     # Define the number of points and create a denser meshgrid
#     num_points = 500  # Increase this value for a denser mesh
#     X_color = np.linspace(min_x - 5, max_x + 5, num_points)
#     Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

#     # Create a meshgrid of points in the X-Y plane
#     X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)

#     # Create an empty array to store class labels for each point in the mesh
#     class_labels = np.zeros(X_mesh.shape)

#     for i in range(X_mesh.shape[0]):
#         for j in range(X_mesh.shape[1]):
#             x_val = X_mesh[i, j]
#             y_val = Y_mesh[i, j]

#             G_color_1 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c1, mean_y_c1], cov[0], prior1)
#             G_color_2 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c2, mean_y_c2], cov[1], prior2)
#             G_color_3 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c3, mean_y_c3], cov[2], prior3)

#             # Find the class with the maximum likelihood
#             max_likelihood_class = np.argmax([G_color_1, G_color_2, G_color_3])

#             # Store the class label in the corresponding location in the array
#             class_labels[i, j] = max_likelihood_class

#     # Now, you can plot the denser mesh with colors based on class labels
#     plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')
#     plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')
#     plt.scatter(X_train_c3, y_train_c3, c='darkgreen', label='Class 3 Training Data')
#     plt.contourf(X_mesh, Y_mesh, class_labels, levels=[0, 0.5, 1.5, 2.5], colors=['r', 'g', 'y'], alpha=0.3)
#     plt.legend()
#     plt.show()



# In[20]:


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler


# #input_data_reshaped = input_data.reshape(-1, 1)

# # Standardizing features for each class separately
# scaler_c1 = StandardScaler()
# X_train_c1 = scaler_c1.fit_transform(X_train_c1)

# scaler_c2 = StandardScaler()
# X_train_c2 = scaler_c2.fit_transform(X_train_c2)

# scaler_c3 = StandardScaler()
# X_train_c3 = scaler_c3.fit_transform(X_train_c3)

# # Creating models for each class separately
# model_c1 = LogisticRegression(solver='liblinear')
# model_c2 = LogisticRegression(solver='liblinear')
# model_c3 = LogisticRegression(solver='liblinear')

# # Fitting models for each class separately
# model_c1.fit(X_train_c1, y_train_c1)
# model_c2.fit(X_train_c2, y_train_c2)
# model_c3.fit(X_train_c3, y_train_c3)

# # Function to plot decision regions
# def plot_decision_regions(models, scalers, titles):
#     plt.figure(figsize=(12, 4))

#     for i, (model, scaler, title) in enumerate(zip(models, scalers, titles), 1):
#         h = 0.02  # Step size in the mesh
#         x_min, x_max = -3, 3
#         y_min, y_max = -3, 3
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#         # Standardize and predict for meshgrid points
#         mesh_points = np.c_[xx.ravel(), yy.ravel()]
#         mesh_points = scaler.transform(mesh_points)
#         Z = model.predict(mesh_points)
#         Z = Z.reshape(xx.shape)

#         plt.subplot(1, 3, i)
#         plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
#         plt.scatter(X_train_c1[:, 0], X_train_c1[:, 1], c=y_train_c1, cmap=plt.cm.Spectral, label='Class C1')
#         plt.scatter(X_train_c2[:, 0], X_train_c2[:, 1], c=y_train_c2, cmap=plt.cm.Spectral, label='Class C2')
#         plt.scatter(X_train_c3[:, 0], X_train_c3[:, 1], c=y_train_c3, cmap=plt.cm.Spectral, label='Class C3')
#         plt.xlabel('Feature 1')
#         plt.ylabel('Feature 2')
#         plt.title(title)
#         plt.legend()

#     plt.tight_layout()
#     plt.show()

# # Plotting decision regions for each class
# models = [model_c1, model_c2, model_c3]
# scalers = [scaler_c1, scaler_c2, scaler_c3]
# titles = ['Decision Boundary for Class C1', 'Decision Boundary for Class C2', 'Decision Boundary for Class C3']

# plot_decision_regions(models, scalers, titles)


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Generating sample data with three classes
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating models: One-vs-Rest (OvR) and One-vs-One (OvO)
ovr_model = LogisticRegression(multi_class='ovr', solver='liblinear')
ovo_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Fitting models
ovr_model.fit(X_train, y_train)
ovo_model.fit(X_train, y_train)

# Function to plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

# Plotting decision boundaries for OvR and OvO models
plot_decision_boundary(ovr_model, X_test, y_test, 'One-vs-Rest (OvR) Decision Boundaries')
plot_decision_boundary(ovo_model, X_test, y_test, 'One-vs-One (OvO) Decision Boundaries')
                       
print(X_train.shape)
print(y_train.shape)
print(y_test.shape)
#print(X_train.type)


# In[22]:


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
plt.title('Decision Regions with Logistic Regression for NLS dataset1')
plt.legend()
plt.show()


# In[23]:


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
plt.title('Decision Regions with Logistic Regression for NLS dataset1')
plt.legend()
plt.show()


# In[24]:


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
plt.title('Decision Regions with Logistic Regression for NLS dataset1')
plt.legend()
plt.show()


# In[25]:


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
plt.title('Decision Regions with Logistic Regression for NLS dataset1')
plt.legend()
plt.show()


# In[42]:


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
plt.title('Decision Regions with KNN (K=1) for NLS dataset1')
plt.legend()
plt.show()


# In[37]:


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
plt.title('Decision Regions with KNN (K=2) for NLS dataset1')
plt.legend()
plt.show()


# In[38]:


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
knn = KNeighborsClassifier(n_neighbors=6)
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
plt.title('Decision Regions with KNN (K=6) for NLS dataset1')
plt.legend()
plt.show()


# In[39]:


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
knn = KNeighborsClassifier(n_neighbors=12)
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
plt.title('Decision Regions with KNN (K=12) for NLS dataset1')
plt.legend()
plt.show()


# In[40]:


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
knn = KNeighborsClassifier(n_neighbors=18)
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
plt.title('Decision Regions with KNN (K=18) for NLS dataset1')
plt.legend()
plt.show()


# In[41]:


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
knn = KNeighborsClassifier(n_neighbors=32)
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
plt.title('Decision Regions with KNN (K=32) for NLS dataset1')
plt.legend()
plt.show()


# In[43]:


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
plt.title('Decision Regions with KNN (K=1) for NLS dataset1')
plt.legend()
plt.show()


# In[44]:


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
plt.title('Decision Regions with KNN (K=2) for NLS dataset1')
plt.legend()
plt.show()


# In[49]:


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
knn = KNeighborsClassifier(n_neighbors=6)
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
plt.title('Decision Regions with KNN (K=6) for NLS dataset1')
plt.legend()
plt.show()


# In[50]:


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
knn = KNeighborsClassifier(n_neighbors=12)
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
plt.title('Decision Regions with KNN (K=12) for NLS dataset1')
plt.legend()
plt.show()


# In[46]:


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
knn = KNeighborsClassifier(n_neighbors=18)
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
plt.title('Decision Regions with KNN (K=18) for NLS dataset1')
plt.legend()
plt.show()


# In[47]:


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
knn = KNeighborsClassifier(n_neighbors=32)
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
plt.title('Decision Regions with KNN (K=32) for NLS dataset1')
plt.legend()
plt.show()


# In[56]:


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
knn = KNeighborsClassifier(n_neighbors=500)
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
plt.title('Decision Regions with KNN (K=500) for NLS dataset1')
plt.legend()
plt.show()


# In[48]:


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
plt.title('Decision Regions with KNN (K=1) for NLS dataset1')
plt.legend()
plt.show()


# In[35]:


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
X_train = np.concatenate(( combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))



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
plt.title('Decision Regions with KNN (K=1) for NLS dataset1')
plt.legend()
plt.show()


# In[58]:


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
X_train = np.concatenate(( combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))



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
plt.title('Decision Regions with KNN (K=1) for NLS dataset1')
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
X_train = np.concatenate(( combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))



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
#plt.scatter(combined_array_train_c1[:, 0], combined_array_train_c1[:, 1], label='Class C1')
plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=2) for NLS dataset1')
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
X_train = np.concatenate(( combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=6)
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
plt.title('Decision Regions with KNN (K=6) for NLS dataset1')
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
X_train = np.concatenate(( combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=12)
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
plt.title('Decision Regions with KNN (K=12) for NLS dataset1')
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
X_train = np.concatenate(( combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=18)
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
plt.title('Decision Regions with KNN (K=18) for NLS dataset1')
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
X_train = np.concatenate(( combined_array_train_c2,  combined_array_train_c3))
y_train = np.concatenate((y_train_c2_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=32)
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
plt.title('Decision Regions with KNN (K=32) for NLS dataset1')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


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
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))



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
plt.title('Decision Regions with KNN (K=1) for NLS dataset1')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


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
knn = KNeighborsClassifier(n_neighbors=6)
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
plt.title('Decision Regions with KNN (K=6) for NLS dataset1')
plt.legend()
plt.show()


# In[29]:


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
knn = KNeighborsClassifier(n_neighbors=6)
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
plt.title('Decision Regions with KNN (K=6) for NLS dataset1')
plt.legend()
plt.show()


# In[30]:


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
knn = KNeighborsClassifier(n_neighbors=32)
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
plt.title('Decision Regions with KNN (K=32) for NLS dataset1')
plt.legend()
plt.show()


# In[31]:


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
knn = KNeighborsClassifier(n_neighbors=32)
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
plt.title('Decision Regions with KNN (K=32) for NLS dataset1')
plt.legend()
plt.show()


# In[32]:


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
knn = KNeighborsClassifier(n_neighbors=32)
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
plt.title('Decision Regions with KNN (K=32) for NLS dataset1')
plt.legend()
plt.show()


# In[33]:


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
knn = KNeighborsClassifier(n_neighbors=32)
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
plt.title('Decision Regions with KNN (K=32) for NLS dataset1')
plt.legend()
plt.show()


# In[59]:


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
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))



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
plt.title('Decision Regions with KNN (K=1) for NLS dataset1')
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
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))



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
#plt.scatter(combined_array_train_c2[:, 0], combined_array_train_c2[:, 1], label='Class C2')
plt.scatter(combined_array_train_c3[:, 0], combined_array_train_c3[:, 1], label='Class C3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions with KNN (K=2) for NLS dataset1')
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
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=6)
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
plt.title('Decision Regions with KNN (K=6) for NLS dataset1')
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
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=12)
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
plt.title('Decision Regions with KNN (K=12) for NLS dataset1')
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
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=18)
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
plt.title('Decision Regions with KNN (K=18) for NLS dataset1')
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
y_train = np.concatenate((y_train_c1_label, y_train_c3_label))



# Creating KNN classifier with K=1
knn = KNeighborsClassifier(n_neighbors=32)
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
plt.title('Decision Regions with KNN (K=32) for NLS dataset1')
plt.legend()
plt.show()


# In[34]:


from sklearn.neighbors import KNeighborsClassifier
# Find the minimum and maximum values for X and Y from your training data
min_x = min(np.min(X_train_c1), np.min(X_train_c2), np.min(X_train_c3))
max_x = max(np.max(X_train_c1), np.max(X_train_c2), np.max(X_train_c3))
min_y = min(np.min(y_train_c1), np.min(y_train_c2), np.min(y_train_c3))
max_y = max(np.max(y_train_c1), np.max(y_train_c2), np.max(y_train_c3))

# Define the number of points and create a denser meshgrid
num_points = 500  # Increase this value for a denser mesh
X_color = np.linspace(min_x - 5, max_x + 5, num_points)
Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

# Create a meshgrid of points in the X-Y plane
X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)

# Create an empty array to store class labels for each point in the mesh
class_labels = np.zeros(X_mesh.shape)

# Combine the training data for all three classes
X_train = np.vstack((X_train_c1, X_train_c2, X_train_c3))
y_train = np.hstack((y_train_c1, y_train_c2, y_train_c3))


# Creating a kNN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Creating a mesh grid to plot decision boundaries
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Predicting the class for each mesh grid point
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundaries
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('kNN Decision Boundary (k=1) for Three Classes')
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate mesh grid
#x_min, x_max = X_train[:, 0].min() - 1, X_train_c1[:, 0].max() + 1
#y_min, y_max = X_train_c1[:, 1].min() - 1, X_train_c1[:, 1].max() + 1
min_x = min(np.min(X_train_c1), np.min(X_train_c2), np.min(X_train_c3))
max_x = max(np.max(X_train_c1), np.max(X_train_c2), np.max(X_train_c3))
min_y = min(np.min(y_train_c1), np.min(y_train_c2), np.min(y_train_c3))
max_y = max(np.max(y_train_c1), np.max(y_train_c2), np.max(y_train_c3))
xx, yy = np.meshgrid(np.arange(min_x, max_x, 0.1),
                     np.arange(min_y, max_y, 0.1))
print(min_x)

# Create kNN models for each class
k = 1  # kNN parameter
knn_c1 = KNeighborsClassifier(n_neighbors=k)
knn_c2 = KNeighborsClassifier(n_neighbors=k)
knn_c3 = KNeighborsClassifier(n_neighbors=k)

knn_c1.fit(X_train_c1.reshape(-1,1),y_train_c1.reshape(-1,1))
knn_c2.fit(X_train_c2.reshape(-1,1),y_train_c2.reshape(-1,1))
knn_c3.fit(X_train_c3.reshape(-1,1),y_train_c3.reshape(-1,1))

# Predict for each mesh point to get the decision boundary
mesh_points = np.c_[xx.ravel(), yy.ravel()]

# Predict for each mesh point to get the decision boundary
Z1 = knn_c1.predict(np.c_[xx.ravel(), yy.ravel()])
#Z1 = Z1.reshape(xx.shape)

Z2 = knn_c2.predict(np.c_[xx.ravel(), yy.ravel()])
#Z2 = Z2.reshape(xx.shape)

Z3 = knn_c3.predict(np.c_[xx.ravel(), yy.ravel()])
#Z3 = Z3.reshape(xx.shape)

# Plotting the decision boundaries
plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z1, alpha=0.4)
plt.contourf(xx, yy, Z2, alpha=0.4)
plt.contourf(xx, yy, Z3, alpha=0.4)

plt.scatter(X_train_c1[:, 0], X_train_c1[:, 1], c='blue', label='Class 1')
plt.scatter(X_train_c2[:, 0], X_train_c2[:, 1], c='red', label='Class 2')
plt.scatter(X_train_c3[:, 0], X_train_c3[:, 1], c='green', label='Class 3')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('kNN Decision Boundaries for k=1')

plt.legend()
plt.show()


# In[ ]:


for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            x_val = X_mesh[i, j]
            y_val = Y_mesh[i, j]
            
            
            

            #G_color_1 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c1, mean_y_c1], cov[0], prior1)
            #G_color_2 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c2, mean_y_c2], cov[1], prior2)
            #G_color_3 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c3, mean_y_c3], cov[2], prior3)

            # Find the class with the maximum likelihood
            min__class = np.argmax([G_color_1, G_color_2, G_color_3])

            # Store the class label in the corresponding location in the array
            class_labels[i, j] = max_likelihood_class

        # Now, you can plot the denser mesh with colors based on class labels
        plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')
        plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')
        plt.scatter(X_train_c3, y_train_c3, c='darkgreen', label='Class 3 Training Data')
        plt.contourf(X_mesh, Y_mesh, class_labels, levels=[0, 0.5, 1.5, 2.5], colors=['r', 'g', 'y'], alpha=0.3)
        plt.legend()
        plt.show()


# In[ ]:


for i, cov in enumerate(covariance_matrices):
# Create a single plot for all three classes with training data and mean
 plt.figure(figsize=(10, 6))

    # Class 1
plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')

    # Class 2
plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')

for i in range(len(X_test_c1)):
        # Calculate log-likelihood for Class 1
        c1 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c1, mean_y_c1], cov[0], prior1)

        # Calculate log-likelihood for Class 2
        c2 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c2, mean_y_c2], cov[1], prior2)

        maxx = max(c1, c2)


# # In[ ]:


# for i, cov in enumerate(covariance_matrices):
#     # Create a single plot for all three classes with training data and mean
#     plt.figure(figsize=(10, 6))

#     # Class 1
#     plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')

#     # Class 2
#     plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')

#     for i in range(len(X_test_c1)):
#         # Calculate log-likelihood for Class 1
#         c1 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c1, mean_y_c1], cov[0], prior1)

#         # Calculate log-likelihood for Class 2
#         c2 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c2, mean_y_c2], cov[1], prior2)

#         maxx = max(c1, c2)

#         if i == 1:
#             if maxx == c1:
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta', label="Class 1 Test Data")
#             else:
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta', label="Class 2 Test Data")
#         else:
#             if maxx == c1:
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta')
#             else:
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta')

#     for i in range(len(X_test_c2)):
#         # Calculate log-likelihood for Class 1
#         c1 = calculate_log_likelihood_single([X_test_c2[i], y_test_c2[i]], [mean_x_c1, mean_y_c1], cov[0], prior1)

#         # Calculate log-likelihood for Class 2
#         c2 = calculate_log_likelihood_single([X_test_c2[i], y_test_c2[i]], [mean_x_c2, mean_y_c2], cov[1], prior2)

#         maxx = max(c1, c2)

#         if i == 1:
#             if maxx == c1:j
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan', label="Class 1 Test Data")
#             else:
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan', label="Class 2 Test Data")
#         else:
#             if maxx == c1:
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan')
#             else:
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan')

#     plt.scatter(mean_x_c1, mean_y_c1, c='black', label='Class 1 Mean')
#     plt.scatter(mean_x_c2, mean_y_c2, c='black', label='Class 2 Mean')
#     plt.title('Training Data and Mean for Two Classes')
#     plt.legend()
#     plt.show()

#     # Find the minimum and maximum values for X and Y from your training data
#     min_x = min(np.min(X_train_c1), np.min(X_train_c2))
#     max_x = max(np.max(X_train_c1), np.max(X_train_c2))
#     min_y = min(np.min(y_train_c1), np.min(y_train_c2))
#     max_y = max(np.max(y_train_c1), np.max(y_train_c2))

#     # Define the number of points and create a denser meshgrid
#     num_points = 500  # Increase this value for a denser mesh
#     X_color = np.linspace(min_x - 5, max_x + 5, num_points)
#     Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

#     # Create a meshgrid of points in the X-Y plane
#     X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)

#     # Create an empty array to store class labels for each point in the mesh
#     class_labels = np.zeros(X_mesh.shape)

#     for i in range(X_mesh.shape[0]):
#         for j in range(X_mesh.shape[1]):
#             x_val = X_mesh[i, j]
#             y_val = Y_mesh[i, j]

#             G_color_1 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c1, mean_y_c1], cov[0], prior1)
#             G_color_2 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c2, mean_y_c2], cov[1], prior2)

#             # Find the class with the maximum likelihood
#             max_likelihood_class = np.argmax([G_color_1, G_color_2])

#             # Store the class label in the corresponding location in the array
#             class_labels[i, j] = max_likelihood_class

#     # Now, you can plot the denser mesh with colors based on class labels
#     plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')
#     plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')
#     plt.contourf(X_mesh, Y_mesh, class_labels, levels=[0, 0.5, 1.5, 2.5], colors=['r', 'g'], alpha=0.3)
#     plt.legend()
#     plt.show()


# # In[ ]:


# for i, cov in enumerate(covariance_matrices):
#     # Create a single plot for all three classes with training data and mean
#     plt.figure(figsize=(10, 6))

#     # Class 1
#     plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')

#     # Class 2
#     plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')

#     for i in range(len(X_test_c1)):
#         # Calculate log-likelihood for Class 1
#         c1 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c1, mean_y_c1], cov[0], prior1)

#         # Calculate log-likelihood for Class 2
#         c2 = calculate_log_likelihood_single([X_test_c1[i], y_test_c1[i]], [mean_x_c2, mean_y_c2], cov[1], prior2)

#         maxx = max(c1, c2)

#         if i == 1:
#             if maxx == c1:
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta', label="Class 1 Test Data")
#             else:
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta', label="Class 2 Test Data")
#         else:
#             if maxx == c1:
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta')
#             else:
#                 plt.scatter(X_test_c1[i], y_test_c1[i], c='magenta')

#     for i in range(len(X_test_c2)):
#         # Calculate log-likelihood for Class 1
#         c1 = calculate_log_likelihood_single([X_test_c2[i], y_test_c2[i]], [mean_x_c1, mean_y_c1], cov[0], prior1)

#         # Calculate log-likelihood for Class 2
#         c2 = calculate_log_likelihood_single([X_test_c2[i], y_test_c2[i]], [mean_x_c2, mean_y_c2], cov[1], prior2)

#         maxx = max(c1, c2)

#         if i == 1:
#             if maxx == c1:
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan', label="Class 1 Test Data")
#             else:
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan', label="Class 2 Test Data")
#         else:
#             if maxx == c1:
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan')
#             else:
#                 plt.scatter(X_test_c2[i], y_test_c2[i], c='cyan')

#     plt.scatter(mean_x_c1, mean_y_c1, c='black', label='Class 1 Mean')
#     plt.scatter(mean_x_c2, mean_y_c2, c='black', label='Class 2 Mean')
#     plt.title('Training Data and Mean for Two Classes')
#     plt.legend()
#     plt.show()

#     # Find the minimum and maximum values for X and Y from your training data
#     min_x = min(np.min(X_train_c1), np.min(X_train_c2))
#     max_x = max(np.max(X_train_c1), np.max(X_train_c2))
#     min_y = min(np.min(y_train_c1), np.min(y_train_c2))
#     max_y = max(np.max(y_train_c1), np.max(y_train_c2))

#     # Define the number of points and create a denser meshgrid
#     num_points = 500  # Increase this value for a denser mesh
#     X_color = np.linspace(min_x - 5, max_x + 5, num_points)
#     Y_color = np.linspace(min_y - 5, max_y + 5, num_points)

#     # Create a meshgrid of points in the X-Y plane
#     X_mesh, Y_mesh = np.meshgrid(X_color, Y_color)

#     # Create an empty array to store class labels for each point in the mesh
#     class_labels = np.zeros(X_mesh.shape)

#     for i in range(X_mesh.shape[0]):
#         for j in range(X_mesh.shape[1]):
#             x_val = X_mesh[i, j]
#             y_val = Y_mesh[i, j]

#             G_color_1 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c1, mean_y_c1], cov[0], prior1)
#             G_color_2 = calculate_log_likelihood_single([x_val, y_val], [mean_x_c2, mean_y_c2], cov[1], prior2)

#             # Find the class with the maximum likelihood
#             max_likelihood_class = np.argmax([G_color_1, G_color_2])

#             # Store the class label in the corresponding location in the array
#             class_labels[i, j] = max_likelihood_class

#     # Now, you can plot the denser mesh with colors based on class labels
#     plt.scatter(X_train_c1, y_train_c1, c='darkred', label='Class 1 Training Data')
#     plt.scatter(X_train_c2, y_train_c2, c='darkblue', label='Class 2 Training Data')
#     plt.contourf(X_mesh, Y_mesh, class_labels, levels=[0, 0.5, 1.5, 2.5], colors=['r', 'g'], alpha=0.3)
#     plt.legend()
#     plt.show()


# # In[ ]:





# # In[ ]:





# # In[ ]:




