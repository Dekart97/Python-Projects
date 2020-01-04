# Author: Dekart Kosa
# Description: This is a linear regression model that predicts the final grade(G3) for students based on mid-term grades, G1,
# G2, study-time, failures, absences. The data is taken from https://archive.ics.uci.edu/ml/datasets/Student+Performance 

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Reading the data from data set
data = pd.read_csv("student-mat.csv", sep=";")

# The the information that is used for the model that we acquired from teh data set
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# The grade that we want the model to predict
predict = "G3"

# The the input for the data
X = np.array(data.drop([predict], 1))
# The output prediction for the data
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# This commented out code is used for training the model
'''best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Linear model
    linear = linear_model.LinearRegression()

    # Linear fit using the training data
    linear.fit(x_train, y_train)
    # Accuracy of the model taken from the training data
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

# Pickle_in stores the trained model and we can re-use it using pickle.load
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient \n', linear.coef_)
print('Intercept \n', linear.intercept_)

# The prediction for G3 taken from looking at the data set and using the linear model.
predictions = linear.predict(x_test)

# Printing all the test predictions
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
