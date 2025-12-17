from provided_code import data_loaders as dl
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class dataset_table_constructor():
    def __init__(self, 
                datasetNames: list[str] = ["balance-scale","balloons","chess-krvk","chess-krvkp","connect-4","contraceptive-method","habermans-survival","hayes-roth","led-display","lymphography","molecular-promoters","molecular-splice","monks-1","monks-2","monks-3","nursery","optdigits","pendigits","semeion","spect-heart","tic-tac-toe","zoo"],
                ):
        self.datasetNames = datasetNames
        self.raw_data = []
        for name in self.datasetNames:
           X, y = dl.load_tabular_xy(f"data/{name}/{name}.data")
           self.raw_data.append((X, y))
    
    def createTable(self):
        headers = [ "Number of Attributes",
                    "Number of Train Cases",
                    "Number of Test Cases",
                    "Number of Classes", 
                    "Class distribution"]
        
        n_attributes = []
        n_train = []
        n_test = []
        n_classes = []
        classDist = []
        totalCases = 0
        totalAttributes = 0

        for (X,y) in self.raw_data:
            trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3)

            totalCases += X.shape[0]
            totalAttributes += X.shape[1]

            # Attributes
            n_attributes.append(X.shape[1])

            # Test/Train
            n_train.append(trainX.shape[0])
            n_test.append(testX.shape[0])

            # Classes
            n_classes.append(len(np.unique(y)))

            # Class Dist    
            classes, counts = np.unique(y, return_counts=True)
            total = len(y)

            dist_str = ", ".join(
                [f"{c}: {cnt} ({(cnt/total)*100:.1f}%)" for c, cnt in zip(classes, counts)]
            )

            classDist.append(dist_str)

        df = pd.DataFrame({
        "Dataset": self.datasetNames,
        "# Attr": n_attributes,
        "Train": n_train,
        "Test": n_test,
        "# Classes": n_classes,
        "Class Distribution": classDist
        })

        latex = df.to_latex(index=False, escape=True)

        print(latex)
        print(totalCases)
        print(totalAttributes)


        



if __name__ == "__main__":
   d = dataset_table_constructor()
   d.createTable()