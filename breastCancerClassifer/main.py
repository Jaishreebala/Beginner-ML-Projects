import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
# from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# Features
x = cancer.data
# Benign / malignant
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clf = svm.SVC(kernel="linear")

# Fit model
clf.fit(x_train, y_train)

# Make predictions
y_pred = clf.predict(x_test)

# Get Accuracy
acc = metrics.accuracy_score(y_test, y_pred)

result = ["malignant", "benign"]
for x in range(len(y_pred)):
    print("Actual Value: ", result[y_test[x]], "Prediction: ", result[y_pred[x]], "\n")
print(acc)