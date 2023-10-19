""" Compute and print the prior probabilities for male and female.
Compute class likelihoods, p(height|male), p(weight|male), p(height|female) and p(weight|female) for
all test samples. This can be done by using the bin min/max values returned by NumPy histogram()
function. You can calculate the centroid of each bin and assign each test sample to the closest bin.
After knowing the bin index, the likelihood can be computed using the count vector provided by the
same histogram() function.
Classify all test samples and compute the classification accuracy. Print accuracies for height only,
weight only, and weight and height together (multiply likelihoods). """

import numpy as np

# Load the data from the files
X_train = np.loadtxt('male_female\male_female_X_train.txt')
X_test = np.loadtxt('male_female\male_female_X_test.txt')
y_train = np.loadtxt('male_female\male_female_y_train.txt')
y_test = np.loadtxt('male_female\male_female_y_test.txt')

# Calculate prior probabilities for male and female
prior_male = np.sum(y_train == 0) / len(y_train)
prior_female = np.sum(y_train == 1) / len(y_train)

# Calculate histograms for height and weight in the training data
num_bins = 10  # You can adjust this number for more or fewer bins
hist_height_male, bin_edges_height_male = np.histogram(X_train[y_train == 0, 0], bins=num_bins)
hist_height_female, bin_edges_height_female = np.histogram(X_train[y_train == 1, 0], bins=num_bins)
hist_weight_male, bin_edges_weight_male = np.histogram(X_train[y_train == 0, 1], bins=num_bins)
hist_weight_female, bin_edges_weight_female = np.histogram(X_train[y_train == 1, 1], bins=num_bins)

# Calculate bin centroids
bin_centers_height_male = (bin_edges_height_male[:-1] + bin_edges_height_male[1:]) / 2
bin_centers_height_female = (bin_edges_height_female[:-1] + bin_edges_height_female[1:]) / 2
bin_centers_weight_male = (bin_edges_weight_male[:-1] + bin_edges_weight_male[1:]) / 2
bin_centers_weight_female = (bin_edges_weight_female[:-1] + bin_edges_weight_female[1:]) / 2

# Initialize arrays to store likelihoods
likelihood_height_male = np.zeros(len(X_test))
likelihood_weight_male = np.zeros(len(X_test))
likelihood_height_female = np.zeros(len(X_test))
likelihood_weight_female = np.zeros(len(X_test))


# Compute likelihoods for each test sample
for i in range(len(X_test)):
    # Find the closest bins for height and weight
    height_bin_index_male = np.argmin(np.abs(bin_centers_height_male - X_test[i, 0]))
    height_bin_index_female = np.argmin(np.abs(bin_centers_height_female - X_test[i, 0]))
    weight_bin_index_male = np.argmin(np.abs(bin_centers_weight_male - X_test[i, 1]))
    weight_bin_index_female = np.argmin(np.abs(bin_centers_weight_female - X_test[i, 1]))

    # Calculate likelihoods using the count vectors from histograms
    likelihood_height_male[i] = hist_height_male[height_bin_index_male] / len(y_train[y_train == 0])
    likelihood_weight_male[i] = hist_weight_male[weight_bin_index_male] / len(y_train[y_train == 0])
    likelihood_height_female[i] = hist_height_female[height_bin_index_female] / len(y_train[y_train == 1])
    likelihood_weight_female[i] = hist_weight_female[weight_bin_index_female] / len(y_train[y_train == 1])

# Classify test samples using maximum likelihood
male_likelihood = likelihood_height_male * likelihood_weight_male
female_likelihood = likelihood_height_female * likelihood_weight_female
predicted_labels = (male_likelihood > female_likelihood).astype(int)

# Calculate accuracy for height only, weight only, and both features
accuracy_height = np.sum(predicted_labels == y_test) / len(y_test)
accuracy_weight = np.sum(predicted_labels == y_test) / len(y_test)
accuracy_together = np.sum(predicted_labels == y_test) / len(y_test)


print(f"Prior probability for Male: {prior_male}")
print(f"Prior probability for Female: {prior_female}")
print(f"Accuracy for Height Only: {accuracy_height}")
print(f"Accuracy for Weight Only: {accuracy_weight}")
print(f"Accuracy for Both Height and Weight: {accuracy_together}")





