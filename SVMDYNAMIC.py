import numpy as np
import urllib
import os
from sklearn import metrics
from sklearn.svm import SVC

# open the source data file
path2 = "G:\dynamicanalyze\dynamicfeauture6.txt"
raw_data= open(path2,'r')
# download the file
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data,delimiter=",")
# separate the data from the target attributes
X = dataset[:,1:49]
y = dataset[:,0]

# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))