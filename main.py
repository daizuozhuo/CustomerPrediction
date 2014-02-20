#!/opt/local/bin/python
import numpy as np
from sklearn.preprocessing import Imputer

TAGMAP = [dict() for i in range(40)]

def file2matrix(filename):
    fr = open(filename)
    fr.readline()
    arrayOLines = fr.readlines()
    rows = len(arrayOLines)
    returnMat = np.zeros((rows, 230))
    returnTag = [[] for i in range(40)]
    index = 0
    #read numerical data into returnMat and Tag into returnTag
    for line in arrayOLines:
        words = line.rstrip('\r\n').split('\t')
        returnMat[index, :190] = map(
            lambda x: float(x) if x.isdigit() else np.nan, words[:190])
        for i in range(40):
            returnTag[i].append(words[190+i])
        index += 1
    
    #define tagmap from tag to number
    for i in range(40):
        s = set(returnTag[i])
        index = 1
        for tag in s:
            TAGMAP[i][tag] = index
            index += 1

    #transform tag into number and add into returnMat 
    for i in range(rows):
        for j in range(40):
            returnMat[i][j] = TAGMAP[j][returnTag[j][i]]
    
    #replcacec missing values into mean of values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(returnMat[:1000])
    returnMat = imp.transform(returnMat)
    return returnMat


if __name__ == "__main__":
    file2matrix("data/orange_small_test.data")
