import math

#Input Example (Peaty Split)
#     Islay   Speyside
# Yes   4        0
# No    1        5
testArr = [[4,0],[1,5]]
testArr2 = [[2,3],[3,2]]
testArr3 = [[2,5],[3,0]]
testArr4 = [[0,0],[0,0]]
testArr5 = [[4,None],[1,5]]
testArr5Target = [[4,0,1],[1,5,0]]

"""
Can take a 2D array of any shape
Handles non number values by puttung them in a missing column
If a category has a probability of 0 it is ignored in the entropy calculator
Returns gain as a float, does not round
Removes rows or columns full of zeroes
"""
def information_gain(table):
    # Make new column for missing values
    table = substituteMissing(table)
    # Remove empty cols or rows
    table = cleanTable(table)       

    # Find the root entropy
    # Create array of value counts
    valueCounts = []
    for col in zip(*table):
        valueCounts.append(sum(list(col)))  
    
    # Convert them to a probability
    probs = []
    for val in valueCounts:
        probs.append(val/sum(valueCounts))

    # Entropy 
    rootEntropy = entropyCalc(probs)

    # Distribution
    dists = []
    for row in table:
        dists.append(sum(row)/sum(valueCounts))   

    # Row Probabilities
    probabilityMatrix = []
    for row in table:
        probRow = []
        for entry in row:
            probRow.append(entry/sum(row))
        probabilityMatrix.append(probRow)    

    # Gain
    nodeEntropies = []
    for x in range(len(dists)):
        nodeEntropies.append(dists[x]*entropyCalc(probabilityMatrix[x]))

    gain = rootEntropy -(sum(nodeEntropies))
    
    return gain

def entropyCalc(probs):
    # Avoid zeroes 
    return -sum(p * math.log2(p) for p in probs if p > 0)

def substituteMissing(table):
    #Go through each row and count missings
    #Append number to each row
    #Replace missings with 0
    for row in table:
        missingCount = 0
        for entry in row:
            if entry is None or (isinstance(entry, float) and math.isnan(entry)):
                missingCount += 1
        row.append(missingCount)

    for row in table:
         for i, entry in enumerate(row):
            if entry is None or (isinstance(entry, float) and math.isnan(entry)):
                row[i] = 0

    return table            

def cleanTable(table):
    # Clean table by removing full 0 rows or columns
    # Clean the rows
    clean_rows = []
    for row in table:
        if any(row): 
            clean_rows.append(row)

    # Clean the columns
    # Flip the rows and columns first
    transposed = list(zip(*clean_rows))

    clean_cols = []
    for col in transposed:
        if any(col):
            clean_cols.append(col)

    # Flip the matrix back
    table = [list(row) for row in zip(*clean_cols)]

    return table

"""
Can take a 2D array of any shape
Handles non number values by putting them in a missing column
If a category has a probability of 0 it is ignored in the entropy calculator
Returns 0 if the split info is 0 to avoid dividing by nothing
Returns gain as a float, does not round
Removes rows or columns full of zeroes
"""
def information_gain_ratio(table):
    # Make new column for missing values
    table = substituteMissing(table)
    # Remove empty cols or rows
    table = cleanTable(table) 
    
    gain = information_gain(table)

    # Get sum of each row
    valueCounts = []
    for col in zip(*table):
        valueCounts.append(sum(list(col)))
    
    # Distribution of each category
    dists = []
    for row in table:
        dists.append(sum(row)/sum(valueCounts))

    splitInfo = -(sum(x*math.log2(x) for x in dists))
    
    return gain/splitInfo if splitInfo != 0 else 0

def square(x):
    return x**2

"""
Can take a 2D array of any shape
Handles non number values by puttung them in a missing column
Removes rows or columns full of zeroes
"""
def chi_squared(table):
    # Make new column for missing values
    table = substituteMissing(table)
    # Remove empty cols or rows
    table = cleanTable(table) 

    # Info needed to make chi annotations
    valueCounts = []
    for col in zip(*table):
        valueCounts.append(sum(list(col)))
    
    probs = []
    for val in valueCounts:
        probs.append(val/sum(valueCounts))

    rowTotals = []
    for row in table:
        rowTotals.append(sum(row))
    
    # Contruct array of (real,predicted) vals
    tuples = []
    for x in range(len(table)):
        for y in range(len(table[0])):
            tuples.append([table[x][y],rowTotals[x]*probs[y]])   

    # Work out formula from tuples
    chi = sum(square(tuples[x][0] - tuples[x][1])/tuples[x][1] for x in range(len(tuples)))

    return chi

"""
Can take a 2D array of any shape
Handles non number values by puttung them in a missing column
Removes rows or columns full of zeroes
Yates Contingency is used if any expected value is less than 0
"""
def chi_squared_yates(table):
    # Make new column for missing values
    table = substituteMissing(table)
    # Remove empty cols or rows
    table = cleanTable(table) 

    # Info needed to make chi annotations
    valueCounts = []
    for col in zip(*table):
        valueCounts.append(sum(list(col)))
    
    probs = []
    for val in valueCounts:
        probs.append(val/sum(valueCounts))

    rowTotals = []
    for row in table:
        rowTotals.append(sum(row))
    
    # Contruct array of (real,predicted) vals
    tuples = []
    for x in range(len(table)):
        for y in range(len(table[0])):
            tuples.append([table[x][y],rowTotals[x]*probs[y]])   

    # Decide whether to do the yates contingency
    lowerThan5 = False
    for x in range(len(tuples)):
        if(tuples[x][1] < 5):
            lowerThan5 = True

    chi = 0
    if not lowerThan5:
       chi = sum(square(tuples[x][0] - tuples[x][1])/tuples[x][1] for x in range(len(tuples)))
    else: 
        chi = sum(square(abs(tuples[x][0] - tuples[x][1]) - 0.5)/tuples[x][1] for x in range(len(tuples)))
    

    return chi

if __name__ == '__main__':
    table = [[4, 0], [1, 5]]
    attribute_name = "Peaty"
    print(f"Table: {table}")


    print(f"information_gain for {attribute_name} = {information_gain(table)}")
    print(f"information_gain_ratio for {attribute_name} = {information_gain_ratio(table)}")
    print(f"chi_squared for {attribute_name} = {chi_squared(table)}")
    print(f"chi_squared_yates for {attribute_name} = {chi_squared_yates(table)}")








    
