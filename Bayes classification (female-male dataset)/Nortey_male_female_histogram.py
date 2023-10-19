""" Compute the histograms of the male height, female height, male weight and female height measurements using the NumPy histogram() function. For the both use 10 bins and fixed ranges ([80, 220]
for height and [30, 180] for weight).
Plot two histograms, one for height and another for weight, that includes the both classes.
Estimate visually which measurement is likely better for classification """

import numpy as np
import matplotlib.pyplot as plt

# Load the data from the text files
male_female_X_test = np.loadtxt('male_female\male_female_X_test.txt')
male_female_X_train = np.loadtxt('male_female\male_female_X_train.txt')
male_female_y_test = np.loadtxt('male_female\male_female_y_test.txt')
male_female_y_train = np.loadtxt('male_female\male_female_y_train.txt')

male_height = male_female_X_train[male_female_y_train == 0][:, 0]
female_height = male_female_X_train[male_female_y_train == 1][:, 0]
male_weight = male_female_X_train[male_female_y_train == 0][:, 1]
female_weight = male_female_X_train[male_female_y_train == 1][:, 1]

number_of_bins = 10
height_range = [80, 220]
weight_range = [30, 180]

male_height_histogram = np.histogram(male_height, bins=number_of_bins, range=height_range)
female_height_histogram= np.histogram(female_height, bins=number_of_bins, range=height_range)
male_weight_histogram = np.histogram(male_weight, bins=number_of_bins, range=weight_range)
female_weight_histogram = np.histogram(female_weight, bins=number_of_bins, range=weight_range)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
plt.hist([male_height, female_height], bins=number_of_bins, range=height_range, label=['Male', 'Female'], alpha=0.7)
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.title('Histogram of Height')
plt.legend()


fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
plt.hist([male_weight, female_weight], bins=number_of_bins, range=weight_range, label=['Male', 'Female'], alpha=0.7)
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.title('Histogram of Weight')
plt.legend()

plt.show()