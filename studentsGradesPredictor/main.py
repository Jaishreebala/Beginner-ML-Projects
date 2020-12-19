import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# Read CSV file
# """
data = pd.read_csv("student-mat.csv", sep=";")

# Select required fields
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Choose predicting field
predict = "G3"

# Separate labels as y and features as X

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Split data into testing and training

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Below code is no longer required as the model is already trained as saved
"""
# Apply linear regression to the data

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# Save predictions in a pickle file

with open("studentgrades.picke", "wb") as f:
    pickle.dump(linear, f)
"""
# Do several predictions and save the best one
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    if linear.score(x_test, y_test) > best:
        best = linear.score(x_test, y_test)
        with open("studentgrades.picke", "wb") as f:
            pickle.dump(linear, f)

# Obtain predictions from the pickle file
pickle_in = open("studentgrades.picke", "rb")
linear = pickle.load(pickle_in)

# Get the accuracy of predictions
acc = linear.score(x_test, y_test)
print(acc)

# Get predictions
predictions = linear.predict(x_test)
print(predictions)

# Loop through predictions and compare with actual data

for x in range(len(predictions)):
    print("Prediction: ", predictions[x], " Dataset: ", x_test[x], "actual value ", y_test[x], "\n")

#  Plot data using matplotlib
plot = "G1"
plt.scatter(data[plot], data["G3"])
# plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grades")
plt.show()
