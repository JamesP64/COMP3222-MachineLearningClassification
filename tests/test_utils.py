import numpy as np

def computeMeasure(pm, prediction, actual):
    if pm == "Accuracy":
        return computeAccuracy(prediction,actual)
    elif pm =="TPR":
        return computeTPR(prediction,actual)
    elif pm == "TNR":
        return computeTNR(prediction,actual)
    elif pm == "Balanced Accuracy":
        return computeBalancedAccuracy(prediction,actual)
    elif pm == "Precision":
        return computePrecision(prediction,actual)
    elif pm == "F1":
        return computeF1(prediction,actual)

def computeAccuracy(y_pred, testy):
    accuracy = np.mean(y_pred == testy) * 100
    return accuracy
    
def computeTPR(y_pred, testy):
    classes = np.unique(testy)
    tprs = []

    # Get the TPR for each class
    for c in classes:
        n_truePositive = np.sum((testy == c) & (y_pred == c))
        n_falseNegative = np.sum((testy == c) & (y_pred != c))
        # Guard against division by 0
        tprs.append(n_truePositive / (n_truePositive+n_falseNegative) if (n_truePositive+n_falseNegative) > 0 else 0.0)

    # Get the average
    return float(np.mean(tprs))
    
def computeTNR(y_pred, testy):
    classes = np.unique(testy)
    tnrs = []

    # Get the TNR for each class
    for c in classes:
        n_trueNegative = np.sum((testy != c) & (y_pred != c))
        n_falsePositive = np.sum((testy != c) & (y_pred == c))
        # Guard against division by 0
        tnrs.append(n_trueNegative / (n_trueNegative + n_falsePositive) if (n_trueNegative + n_falsePositive) > 0 else 0.0)

    return float(np.mean(tnrs))

def computeBalancedAccuracy(y_pred, testy):
    tpr = computeTPR(y_pred, testy)
    tnr = computeTNR(y_pred, testy)
        
    return ((tpr+tnr)/2)
    
def computePrecision(y_pred, testy):
    classes = np.unique(testy)
    precisions = []

    # Get the Precision for each class    
    for c in classes:
        n_truePositive = np.sum((testy == c) & (y_pred == c))
        n_falsePositive = np.sum((testy != c) & (y_pred == c))
        # Guard against division by 0
        precisions.append(n_truePositive / (n_truePositive + n_falsePositive) if (n_truePositive + n_falsePositive) > 0 else 0.0)

    return float(np.mean(precisions))
    
def computeF1(y_pred, testy):
    precision = computePrecision(y_pred,testy)
    recall = computeTPR(y_pred,testy)
    return float(2*((precision*recall)/(precision+recall)))