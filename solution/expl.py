import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s"
)

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata

X = np.array([
    ["yes", "no",  "yes"],
    ["yes", "yes", "yes"],
    ["yes", "no",  "no"],
    ["yes", "no",  "no"],
    ["no",  "yes", "no"],
    ["no",  "yes", "yes"],
    ["no",  "yes", "yes"],
    ["no",  "yes", "yes"],
    ["no",  "no",  "yes"],
    ["no",  "no",  "yes"]
])

y = np.array([
    "Islay",
    "Islay",
    "Islay",
    "Islay",
    "Islay",
    "Speyside",
    "Speyside",
    "Speyside",
    "Speyside",
    "Speyside"
])

# X,y = dl.load_tabular_xy(f"data/lymphography/lymphography.data")
# trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3, random_state = 42)
# clf = te.TreeEnsembleClassifier(n_estimators=2)
# clf.fit(trainX,trainy)
# y_pred = clf.predict(trainX)

# print(f"Accuracy: {np.mean(y_pred == trainy) * 100}") 
differences = np.array([1.6,1.2,1.3,1.4])
abs_differences = np.abs(differences)
ranks = rankdata(abs_differences, method='average')
print(ranks)