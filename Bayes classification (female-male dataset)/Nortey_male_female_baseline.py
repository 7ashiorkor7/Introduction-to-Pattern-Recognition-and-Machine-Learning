""" Make a classifier that assigns a random class to each test sample and computes its classification
accuracy. Print the accuracy.
Make another classifier that assigns the most likely class (highest a priori) to all test samples. Print
the accuracy """

import random
import numpy as np

male_female_X_test = np.loadtxt('male_female\male_female_X_test.txt')
male_female_X_train = np.loadtxt('male_female\male_female_X_train.txt')
male_female_y_test = np.loadtxt('male_female\male_female_y_test.txt')
male_female_y_train = np.loadtxt('male_female\male_female_y_train.txt')

# Classifier that assigns a random class to each test sample
def random_classifier(male_female_X_test):
    return [random.choice([0, 1]) for _ in range(len(male_female_X_test))]

# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Calculate Accuracy for Classifier that assigns a random class to each test sample
random_predictions = random_classifier(male_female_X_test)
random_accuracy = calculate_accuracy(male_female_y_test, random_predictions)
print("Random Classifier Accuracy:", random_accuracy)


# Classifier that assigns the most likely vlass to all test samples
def mostlikely_classifier(X_test, y_train):
    male_count = sum(1 for label in y_train if label == 0)
    female_count = len(y_train) - male_count
    return [0 if male_count > female_count else 1 for _ in range(len(X_test))]


# Accuracy for Classifier that assigns the most likely class to all test samples
mostlikely_predictions = mostlikely_classifier(male_female_X_test, male_female_y_train)
mostlikely_accuracy = calculate_accuracy(male_female_y_test, mostlikely_predictions)
print("Most Likely Classifier Accuracy:", mostlikely_accuracy)

