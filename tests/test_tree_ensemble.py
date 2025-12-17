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
        #datasetNames: list[str] = ["balance-scale","balloons","chess-krvk","chess-krvkp","connect-4","contraceptive-method","habermans-survival","hayes-roth","led-display","lymphography","molecular-promoters","molecular-splice","monks-1","monks-2","monks-3","nursery","optdigits","pendigits","semeion","spect-heart","tic-tac-toe","zoo"],
        datasetNames: list[str] = ["pendigits"],
        testSplitRatio: float = 0.3,
        n_estimators: int = 50,
        n_iterations: int = 10
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

        # Create random seeds
        randomSeeds = np.arange(self.n_iterations)

        # Run iterations of test
        results = Parallel(n_jobs=4, backend="loky")(
        delayed(self._run_single_iteration)(seed)
        for seed in randomSeeds
        )

        # Unpack results
        averageResultsMatrix = []
        wilcoxonIterations = []
        for Wresult, averagedResults in results:
            averageResultsMatrix.append(averagedResults)
            wilcoxonIterations.append((Wresult.statistic, Wresult.pvalue))
        print("\nAll iterations complete.") 

        # Store Stats CSV
        table_graph_builders.statsTable(averageResultsMatrix,self.compMode)

        # Wilcoxon CSVs and Graph
        table_graph_builders.wilcoxonTable(wilcoxonIterations,self.compMode)

        # Stats Bar Chart
        table_graph_builders.performanceMeasuresBarChart(averageResultsMatrix, self.compMode)

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
            print(self.datasetNames[i])
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
                if pm == "Balanced Accuracy":
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
    
    def compare_all_three(self):
        print("Starting 3-Way Comparison: Original vs RF vs LR...")
        
        # Run iterations in parallel
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self._run_all_three_on_datasets)(seed)
            for seed in np.arange(self.n_iterations)
        )
        
        # Placeholders for final averages
        final_stats = []
        
        for i, pm in enumerate(self.performanceMeasures):
            # Extract the i-th metric from all iterations
            orig_vals = [row[i][1] for row in results]
            rf_vals   = [row[i][2] for row in results]
            lr_vals   = [row[i][3] for row in results]
            
            # Calculate means
            mean_orig = np.mean(orig_vals)
            mean_rf   = np.mean(rf_vals)
            mean_lr   = np.mean(lr_vals)

            # --- SCALE ACCURACY BY 100 ---
            # If your utils return 65.0, this converts it to 0.65
            if pm == "Accuracy":
                mean_orig /= 100.0
                mean_rf   /= 100.0
                mean_lr   /= 100.0

            final_stats.append({
                "Metric": pm,
                "Original": mean_orig,
                "Random Forest": mean_rf,
                "Logistic Regression": mean_lr
            })

        print("Comparison complete. Generating graph...")
        table_graph_builders.plot_three_way_bar(final_stats)

    def _run_all_three_on_datasets(self, seed):
        self.randomState = seed
        
        # Accumulators for this specific seed (iteration)
        cum_orig = np.zeros(len(self.performanceMeasures))
        cum_rf   = np.zeros(len(self.performanceMeasures))
        cum_lr   = np.zeros(len(self.performanceMeasures))

        for i in range(self.n_datasets):
            # Load Data
            X, y = self.raw_data[i]
            X_ohe, y_ohe = self.ohe_data[i]

            # Split (same seed ensures fair comparison)
            trainX, testX, trainy, testy = train_test_split(X, y, test_size=self.testSplitRatio, random_state=seed)
            ohetrainX, ohetestX, ohetrainy, ohetesty = train_test_split(X_ohe, y_ohe, test_size=self.testSplitRatio, random_state=seed)

            # 1. Original Custom Tree
            clf = te.TreeEnsembleClassifier(n_estimators=self.n_estimators, random_state=seed)
            clf.fit(trainX, trainy)
            pred_orig = clf.predict(testX)

            # 2. Random Forest (sklearn)
            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(ohetrainX, ohetrainy)
            pred_rf = rf.predict(ohetestX)

            # 3. Logistic Regression (sklearn)
            lr = LogisticRegression(max_iter=2000, n_jobs=1, random_state=seed)
            lr.fit(ohetrainX, ohetrainy)
            pred_lr = lr.predict(ohetestX)

            # Calculate Metrics
            for idx, pm in enumerate(self.performanceMeasures):
                cum_orig[idx] += test_utils.computeMeasure(pm, pred_orig, testy)
                cum_rf[idx]   += test_utils.computeMeasure(pm, pred_rf, ohetesty)
                cum_lr[idx]   += test_utils.computeMeasure(pm, pred_lr, ohetesty)

        # Average over datasets for this iteration
        iter_results = []
        for idx, pm in enumerate(self.performanceMeasures):
            iter_results.append((
                pm,
                cum_orig[idx] / self.n_datasets,
                cum_rf[idx]   / self.n_datasets,
                cum_lr[idx]   / self.n_datasets
            ))
            
        return iter_results

if __name__ == "__main__":
    test = TestTreeEnsemble()
    test.testExperiment(0)





    
