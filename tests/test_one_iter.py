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
from tests import table_graph_builders

class TestTreeEnsemble():

    def __init__(
        self,
        datasetNames: list[str] = ["balance-scale","balloons","chess-krvk","chess-krvkp","connect-4","contraceptive-method","habermans-survival","hayes-roth","led-display","lymphography","molecular-promoters","molecular-splice","monks-1","monks-2","monks-3","nursery","optdigits","pendigits","semeion","spect-heart","tic-tac-toe","zoo"],
        testSplitRatio: float = 0.3,
        n_estimators: int = 50,
        n_iterations: int = 1
    ):

        # datasetNames controls the datasets used
        self.datasetNames = datasetNames
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
        # Number of datasets
        self.n_datasets = len(datasetNames)

        # Preload the datasets (X and y)
        self.raw_data = [] 
        self.ohe_data = []   

        for name in self.datasetNames:
           X, y = dl.load_tabular_xy(f"data/{name}/{name}.data")
           self.raw_data.append((X, y))

           ohe = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
           X_ohe = ohe.fit_transform(X)
           self.ohe_data.append((X_ohe, y))

    def testExperiment(self, compIndex):
        # Store chosen model 
        self.compMode = self.compNames[compIndex]

        # Run iterations of test
        self._run_single_iteration(5)

    def _run_single_iteration(self, seed):
        print(f"Iteration seed={seed}")
        self.randomState = seed
        return self.runOnDatasets()
        
    def runOnDatasets(self):
        # Keep track of cumulative stats for averaged table
        cumulativeOriginalStats = np.zeros(len(self.performanceMeasures))
        cumulativeOHEStats = np.zeros(len(self.performanceMeasures))
        # Keep track of individual Baccs for Wilcoxon
        original_Baccs = []
        ohe_Baccs = []

        # Start Comparison
        for i in range(self.n_datasets):
            results = []

            # Get data
            X, y = self.raw_data[i]
            X_ohe, y_ohe = self.ohe_data[i]
            # Split data
            trainX, testX, trainy, testy = train_test_split(X, y, test_size = self.testSplitRatio, random_state = self.randomState)
            ohetrainX, ohetestX, ohetrainy, ohetesty = train_test_split(X_ohe, y_ohe, test_size = self.testSplitRatio, random_state = self.randomState)

            # Fit and Predict Original
            clf = te.TreeEnsembleClassifier(n_estimators=self.n_estimators, random_state = self.randomState)
            clf.fit(trainX,trainy)
            y_pred = clf.predict(testX)

            # Fit and Predict Competior
            compClf = {
                        "OHE": lambda: te.TreeEnsembleClassifier(n_estimators=self.n_estimators, random_state=self.randomState),
                        "Random Forest": lambda: RandomForestClassifier(random_state=self.randomState),
                        "Logistic Regression": lambda: LogisticRegression(max_iter=2000, n_jobs=-1)
            }[self.compMode]()
            compClf.fit(ohetrainX,ohetrainy)
            compy_pred = compClf.predict(ohetestX)

            # Compute performance measure
            for j,pm in enumerate(self.performanceMeasures):
                # Compute and store performance
                orig = test_utils.computeMeasure(pm, y_pred, testy)
                comp = test_utils.computeMeasure(pm, compy_pred, ohetesty)
                cumulativeOriginalStats[j] += orig
                cumulativeOHEStats[j] += comp
                results.append((pm,orig,comp))

                # Store diff for Wilcoxon if Bacc
                if pm == "Accuracy":
                    print(f"Dataset: {self.datasetNames[i]} , Accuracy: {orig}")
                    print(f"Dataset: {self.datasetNames[i]} , Accuracy OHE : {comp}")

                if pm == "Balanced Accuracy":
                    print(f"Dataset: {self.datasetNames[i]} , Balanced Accuracy: {orig}")
                    print(f"Dataset: {self.datasetNames[i]} , Balanced Accuracy OHE : {comp}")
                    diff = orig-comp
                    # Ignore no diffs
                    if not diff == 0:
                        original_Baccs.append(orig)
                        ohe_Baccs.append(comp)
            
        # Averaged results
        averagedResults = []
        for k,measure in enumerate(self.performanceMeasures):
                averagedResults.append((measure, (cumulativeOriginalStats[k]/self.n_datasets), (cumulativeOHEStats[k]/self.n_datasets)))

        # Wilcoxon test
        Wresult = wilcoxon(original_Baccs, ohe_Baccs, alternative="two-sided")

        return Wresult, averagedResults

if __name__ == "__main__":
    test = TestTreeEnsemble()
    test.testExperiment(0)





    
