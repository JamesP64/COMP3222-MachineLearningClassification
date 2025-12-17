import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import solution.utils as util
import solution.decision_stump
import random
import math
from provided_code import data_loaders as dl
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class TreeEnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_estimators: int = 200,
        average_probas: bool = True,
        random_state: int | None = None
    ):
        
        # n estimators controls ensemble size
        self.n_estimators = n_estimators
        # Average probas switches between majority vote (hard voting) and averaging predicted probabilities (soft voting)
        self.average_probas = average_probas
        # Seed for rng
        self.random_state = random_state

        # Array of all stumps with their training X and y
        self.estimators = []
        # The predictions from each estimator
        self.estimatorPredictions = []
        # Map of unique class lables
        self.uniqueClasses = None
        # Number of cases in the dataset
        self.casesCount = None
        # Number of features in the dataset
        self.featuresCount =  None
        # Map of what estimator sees which features
        self.featureIndicies = [[] for i in range(self.n_estimators)]
        # FOR CASE STUDY: Map of which attribute each stump split on
        self.AttrsSplitOn = []
    
    """
    Contruct classifiers and fit each one
    Input:
        X: numpy array (n_cases, n_attributes)
        The input data
        
        y: numpy array (n_cases,)
        The target variables
    Diversity Strategy:
        Sampling of size n by case with replacement
        Each stump sees random subspace of features
        Randomising attribute quality
    """
    def fit(self, X, y):
        # Set the random seed
        if self.random_state is not None:
            random.seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        # Input has to be an object so its not ints or floats or anything
        if X.dtype.kind != 'O':
            X = X.astype(object)

        # Check for floats hiding as strings
        if util.is_real_number_array(X):
            raise TypeError("No real-valued attributes can be present in X")

        # Convert all values to strings or "MISSING"
        X = util.fast_normalize(X) 
                
        # Map of unique classes
        self.uniqueClasses = np.unique(y)
        # Number of cases in the dataset
        self.casesCount = len(X)
        # Number of features in the dataset
        self.featuresCount =  X.shape[1]
        # Size of random subspace for each stump
        k = round(math.sqrt(self.featuresCount))
        k = self.featuresCount

        # Resetting estimator arrays
        self.estimators = []
        self.featureIndicies = [[] for _ in range(self.n_estimators)]

        # Create each estimator, store with its X and y
        for i in range(self.n_estimators):
            # Create a random subspace size k
            randomColumns = random.sample(range(self.featuresCount),k)

            # Store which subspace this one gets
            self.featureIndicies[i] = randomColumns

            # Samples of size n for n cases (with replacement)
            randomRows = random.choices(range(self.casesCount), k=self.casesCount)

            # New X for this estimator
            X_sample = X[randomRows, :][:, randomColumns]
            # New y for this sample
            y_sample = y[randomRows]

            # Random quality measure
            measures = ["ig", "gain ratio", "chi2", "chi2 yates"]
            # pick a single measure as a string
            randomMeasure = random.choice(measures)

            # Create the classifier
            clf = solution.decision_stump.DecisionStumpClassifier(n_attributes=k,quality_measure=randomMeasure,random_state=self.random_state)
            self.estimators.append([clf,X_sample,y_sample])
        
        # fit each estimator
        for estimator in self.estimators:
            estimator[0].fit(estimator[1],estimator[2])
            self.AttrsSplitOn.append(randomColumns[estimator[0].att_index])
    """
    Call predict on each classifier
    Collect hard votes over uniqueClasses    
    """
    def predict(self, X):
        # Convert all values to strings or "MISSING"
        X = util.fast_normalize(X)

        self.estimatorPredictions = []

        # Hard Voting
        if not self.average_probas:
            # Get each stump's predictions set
            for i,estimator in enumerate(self.estimators):
                # The subset of features that this estimator sees
                cols = self.featureIndicies[i]
                self.estimatorPredictions.append(estimator[0].predict(X[:, cols]))

            preds = np.array(self.estimatorPredictions)
            n_cases = X.shape[0]

            # Need votes distribution for each case in X
            votesForEachCase = []
            for case_idx in range(n_cases):
                casePredictions = preds[:, case_idx]
                # Count votes for each class
                votes = [(casePredictions == c).sum() for c in self.uniqueClasses]
                votesForEachCase.append(votes)

            votesForEachCase = np.array(votesForEachCase)
            
            # Decide final prediction for each case, lower index in classes breaks ties     
            # Find the most voted class label for each row
            idx = np.argmax(votesForEachCase, axis=1) 
            # Convert that to an actual class label
            return self.uniqueClasses[idx]
        # Soft voting, argmax of predict_proba
        else:
            # Get the probability matrix
            proba = self.predict_proba(X)     
            # Find the most likely class label for each row
            idx = np.argmax(proba, axis=1) 
            # Convert that to an actual class label
            return self.uniqueClasses[idx]

    """
    1. If average probas=True, obtain each base classifier’s predict proba, align columns
    to classes , average across members, and return an (n samples × n classes) array whose rows sum to 1.
    
    2. If average probas=False, convert hard votes into class counts per case and divide
    by n estimators to yield vote proportions (rows sum to 1).

    Input:
        X: numpy array of shape (n_cases, n_attributes)
        The dataset to evaluate. Each stump uses the feature subset it was
        trained on.

    Output:
        avg_proba: numpy array of shape (n_cases, n_uniqueClasses)
        The averaged probability distribution over classes for each case.
        - Rows correspond to cases in X.
        - Columns correspond to classes in self.uniqueClasses.
        - Rows are normalised to sum to 1.

    """
    def predict_proba(self, X):
        # Convert all values to strings or "MISSING"
        X = util.fast_normalize(X)

        n_cases = X.shape[0]
        n_uniqueClasses = len(self.uniqueClasses)

        # Estimator makes a prediction matrix: the chance of each attribute value producing each class label
        all_probas = []
        for i,estimator in enumerate(self.estimators):
            # Get the subset of features that this estimator sees
            X_subset = X[:, self.featureIndicies[i]]
            # Get the prediction matrix from the estimator
            estimator_probas = estimator[0].predict_proba(X_subset)

            # Slot in stump's probabilities into the overall matrix
            fullMatrix = np.zeros((n_cases, n_uniqueClasses))
            for j, c in enumerate(estimator[0].classes_):
                # Find the right class label and slot the probability row into the full matrix
                class_idx = np.where(self.uniqueClasses == c)[0][0]
                fullMatrix[:, class_idx] = estimator_probas[:, j]

            # Store an array of aligned matricess
            all_probas.append(fullMatrix)

        # Get the mean of all matrixes to create one matrix of probabilitis
        avg_proba = np.mean(all_probas, axis=0)

        # Normalise rows so they sum to 1 
        avg_proba /= avg_proba.sum(axis=1, keepdims=True)

        return avg_proba
            
if __name__ == "__main__":
    import numpy as np
    from provided_code import data_loaders as dl
    from sklearn.model_selection import train_test_split

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

    # Load Data
    X, y = dl.load_tabular_xy("data/fertility/fertility.data")

    # Split it
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the ensemble and train it
    clf = TreeEnsembleClassifier(random_state=42)
    clf.fit(trainX,trainy)

    # Make a prediction on the training set
    y_pred = clf.predict(trainX)

    # Compute training accuracy
    accuracy = np.mean(y_pred == trainy) * 100

    # Print result
    print(f"Tree ensemble on fertility problem has accuracy {accuracy:.2f}%")




