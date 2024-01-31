import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

np.random.seed(4)

# Function to extract 24-dimensional color histogram feature vectors from an image
def extract_color_histogram(image):
    histograms = []
    for i in range(0, image.shape[0], 32):
        for j in range(0, image.shape[1], 32):
            patch = image[i:i+32, j:j+32]

            # Calculate color histograms for each channel (R, G, B)
            hist_r = cv2.calcHist([patch], [0], None, [8], [0, 256])
            hist_g = cv2.calcHist([patch], [1], None, [8], [0, 256])
            hist_b = cv2.calcHist([patch], [2], None, [8], [0, 256])

            # Concatenate the histograms to form a 24-dimensional feature vector
            feature_vector = np.concatenate((hist_r, hist_g, hist_b)).flatten()
            histograms.append(feature_vector)

    return histograms

# Function to extract BoVW representation from an image
def extract_bovw_representation(image, kmeans):
    histograms = extract_color_histogram(image)

    # Assign each 24-dimensional color histogram feature vector to a cluster
    labels = kmeans.predict(histograms)

    # Count the number of feature vectors assigned to each cluster
    bovw_representation, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))

    # Normalize the vector
    bovw_representation = bovw_representation / len(histograms)

    return bovw_representation

# Directory paths
train_directory = "group18-2/train/"
test_directory = "group18-2/test/"

# Initialize lists to store BoVW representations and labels for both training and test data
train_bovw_data = []
train_bovw_labels = []
test_bovw_data = []
test_bovw_labels = []

# Use K-means clustering to create the Bag-of-Visual-Words (BoVW) representation
kmeans = KMeans(n_clusters=32)

# Extract color histograms for training data
train_histograms = []
for class_folder in os.listdir(train_directory):
    class_path = os.path.join(train_directory, class_folder)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            # Read the image
            img = cv2.imread(image_path)

            # Extract color histograms
            histograms = extract_color_histogram(img)
            train_histograms.extend(histograms)

# Ensure data type consistency
train_histograms = np.array(train_histograms, dtype=np.float64)

# Fit the KMeans model to the color histograms
kmeans.fit(train_histograms)

# Extract BoVW representations and labels for training data
for class_folder in os.listdir(train_directory):
    class_path = os.path.join(train_directory, class_folder)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            # Read the image
            img = cv2.imread(image_path)

            # Extract BoVW representation
            bovw_representation = extract_bovw_representation(img, kmeans)

            # Append the BoVW representation to the training data
            train_bovw_data.append(bovw_representation)
            train_bovw_labels.append(class_folder)

# Convert the lists to NumPy arrays for training data
train_bovw_data = np.array(train_bovw_data)
train_bovw_labels = np.array(train_bovw_labels)

# Save the BoVW representations and labels to NumPy files for training data
np.save("./data/train_bovw_data.npy", train_bovw_data)
np.save("./data/train_bovw_labels.npy", train_bovw_labels)

# Extract BoVW representations and labels for test data
for class_folder in os.listdir(test_directory):
    class_path = os.path.join(test_directory, class_folder)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            # Read the image
            img = cv2.imread(image_path)

            # Extract BoVW representation
            bovw_representation = extract_bovw_representation(img, kmeans)

            # Append the BoVW representation to the test data
            test_bovw_data.append(bovw_representation)
            test_bovw_labels.append(class_folder)

# Convert the lists to NumPy arrays for test data
test_bovw_data = np.array(test_bovw_data)
test_bovw_labels = np.array(test_bovw_labels)

# Save the BoVW representations and labels to NumPy files for test data
np.save("./data/test_bovw_data.npy", test_bovw_data)
np.save("./data/test_bovw_labels.npy", test_bovw_labels)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load BoVW representations
train_bovw_data = np.load("./data/train_bovw_data.npy")
test_bovw_data = np.load("./data/test_bovw_data.npy")

# Load true labels as strings
true_labels_str = np.load("./data/test_bovw_labels.npy")

maxx_acc = 0
gmm_mixtures = 0
pca_components = 0

accuracy_1 = []
accuracy_2 = []
accuracy_4 = []
accuracy_8 = []

# Convert true labels to numeric format using LabelEncoder
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(true_labels_str)
true_labels = np.array(true_labels)

num_classes = np.unique(true_labels)

# Class labels
class_labels = label_encoder.classes_

# Loop through different values of PCA dimensions (l)
for n_components_pca in range(32, 0, -1):
    # Perform PCA
    pca = PCA(n_components=n_components_pca)
    train_bovw_reduced = pca.fit_transform(train_bovw_data)
    test_bovw_reduced = pca.transform(test_bovw_data)

    # Plot eigenvalues
    plt.figure(figsize=(8, 4))
    plt.scatter(np.arange(1, n_components_pca + 1), pca.explained_variance_, marker='o')
    plt.title(f'Eigenvalues for PCA={n_components_pca}')
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalues')
    plt.grid(True)
    plt.show()

    # Plot 2D scatter plot for only PCA dimesions (l) = 2
    if n_components_pca == 2:
        plt.figure(figsize=(8, 8))
        plt.scatter(train_bovw_reduced[:, 0], train_bovw_reduced[:, 1], c=true_labels, cmap='tab10')
        plt.title(f'PCA={n_components_pca}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()

    # Loop through different numbers of mixtures
    for n_components_gmm in [1, 2, 4, 8]:
        # Build GMM
        gmm = GaussianMixture(n_components=n_components_gmm)
        gmm.fit(train_bovw_reduced)

        # Evaluate on test data
        predictions = gmm.predict(test_bovw_reduced)

        # Calculate and print performance metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=1)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=1)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=1)
        confusion_mat = confusion_matrix(true_labels, predictions, labels=num_classes)

        if maxx_acc < accuracy:
            maxx_acc = accuracy
            gmm_mixtures = n_components_gmm
            pca_components = n_components_pca

        # Display results
        print(f"PCA: {n_components_pca}, GMM Mixtures: {n_components_gmm}")
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        print("Confusion Matrix:")
        print(confusion_mat)

        if n_components_gmm == 1:
            accuracy_1.append(accuracy)
        elif n_components_gmm == 2:
            accuracy_2.append(accuracy)
        elif n_components_gmm == 4:
            accuracy_4.append(accuracy)
        else:
            accuracy_8.append(accuracy)

        # Plot confusion matrix as a heatmap with class labels
        # figure size 2*2
        plt.figure(figsize=(2, 2))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Oranges', cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix for PCA={n_components_pca}, GMM Mixtures={n_components_gmm}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

print ("Max Accuracy: ", maxx_acc)
print ("GMM Mixtures: ", gmm_mixtures)
print ("PCA Components: ", pca_components)


# Plot accuracy vs number of PCA dimensions as a bar chart
plt.figure(figsize=(10, 6))

# reverse the lists to match the order of the PCA dimensions
accuracy_1.reverse()
accuracy_2.reverse()
accuracy_4.reverse()
accuracy_8.reverse()

bar_width = 0.2  # Width of each bar
bar_positions = np.arange(1, 33)

plt.bar(bar_positions - bar_width, accuracy_1, label='GMM Mixtures=1', width=bar_width, color='blue')
plt.bar(bar_positions, accuracy_2, label='GMM Mixtures=2', width=bar_width, color='orange')
plt.bar(bar_positions + bar_width, accuracy_4, label='GMM Mixtures=4', width=bar_width, color='green')
plt.bar(bar_positions + 2 * bar_width, accuracy_8, label='GMM Mixtures=8', width=bar_width, color='red')

plt.title('Accuracy vs PCA Dimensions')
plt.xlabel('PCA Dimensions')
plt.ylabel('Accuracy')
plt.xticks(bar_positions, np.arange(1, 33))  # Set x-axis ticks to match the PCA dimensions
plt.legend()
plt.grid(axis='y')  # Add grid lines on the y-axis for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()