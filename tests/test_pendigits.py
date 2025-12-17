import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from solution import tree_ensemble as te
from provided_code import data_loaders as dl
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import rankdata
from scipy.stats import wilcoxon
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tests import test_utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TestPendigits():

    def __init__(
        self,
        datasetName: str = "pendigits",
        testSplitRatio: float = 0.3,
        n_estimators: int = 50,
        n_iterations: int = 10
    ):

        # datasetNames controls the datasets used
        self.datasetName = datasetName
        # testSplitRatio determine how much data is used for training/testing
        self.testSplitRatio = testSplitRatio
        # n estimators controls ensemble size 
        self.n_estimators = n_estimators
        # n iterations determines how many tests are repeated with different random states
        self.n_iterations = n_iterations

        # Possible comparisons
        self.compNames = ["OHE", "Random Forest", "Logistic Regression"]
        # Chosen model to compare against
        self.compMode = None
        # Evaluation Methods
        self.performanceMeasures = ["Accuracy", "TPR", "TNR", "Balanced Accuracy", "Precision", "F1"]

        # Preload the datasets (X and y)
        self.raw_data = [] 
        self.ohe_data = []   

        X, y = dl.load_tabular_xy(f"data/pendigits/pendigits.data")
        self.raw_data.append((X, y))

        ohe = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
        X_ohe = ohe.fit_transform(X)
        self.ohe_data.append((X_ohe, y))

    def getAttributesPicked(self):
        # Get data
        X, y = self.raw_data[0]
        # Split data
        trainX, testX, trainy, testy = train_test_split(X, y, test_size = self.testSplitRatio, random_state = 42)
        
        clf = te.TreeEnsembleClassifier(n_estimators=self.n_estimators, random_state = 42)
        clf.fit(trainX,trainy)
        attrsSplitOn = clf.AttrsSplitOn
        distArr = np.zeros(16)
        for i in attrsSplitOn:
            distArr[i] += 1

    
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(distArr)), distArr, color='skyblue', edgecolor='black')
        plt.xlabel('Attribute Index')
        plt.ylabel('Count')
        plt.title('Attribute Split Distribution')
        plt.xticks(range(len(distArr)))
        plt.show()

    def printShapes(self):
        # Get data
        X, y = self.raw_data[0]
        ohe_X, y = self.ohe_data[0]
        print(X.shape)
        print(ohe_X.shape)


if __name__ == "__main__":
    test = TestPendigits()
    test.compare("OHE")     