#!/opt/local/bin/python
from numpy import *

def file2matrix(filename):
    fr = open(filename)
    fr.readline()
    arrayOLines = fr.readlines()
    rows = len(arrayOLines)
    returnMat = zeros((rows, 190))
    returnTag = []
    index = 0
    for line in arrayOLines:
        words = line.rstrip('\n').split('\t')
        #to do: replace missing value
        returnMat[index, :] = map(lambda x: x if x.isdigit() else -1, 
                                  words[:190])
        returnTag.append(words[190:])
        index += 1
        #print returnMat[0], returnTag[0]
    return returnMat, returnTag

if __name__ == "__main__":
    file2matrix("data/orange_small_test.data")
