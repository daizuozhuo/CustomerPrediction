#!/opt/local/bin/python
from collections import Counter
import itertools
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


def file2binMatrix(filename):
    fr = open(filename)
    fr.readline()
    arrayOLines = fr.readlines()

    catags = [[] for i in range(40)]
    data = []
    for line in arrayOLines:
        line = line.rstrip('\r\n').split('\t')
        line[:190] = map(
            lambda x: float(x) if x.isdigit() else np.nan, line[:190])
        data.append(line)
        for i in range(40):
            catags[i].append(line[190+i])

    #only encode the 10 most common values for each catagorical feature
    keys= [[] for i in range(40)]
    for i in range(40):
        c = Counter(catags[i])
        for key, number in c.most_common(10):
            keys[i].append(key)
    #replace other values to empty
    returnMat = []
    for line in data:
        for i in range(40):
            if line[190+i] in keys[i]:
                line[190+i] = keys[i].index(line[190+i])
            else:
                line[190+i] = 10
        returnMat.append(line)
 
    '''
    measure = []
    for line in data:
        for i in range(40):
            if line[190+i] in keys[i]:
                line[190+i] = keys[i].index(line[190+i])
            else:
                line[190+i] = 10
        measure.append(dict( zip(range(len(line)), line) ))

    #use DictVectorizer to process catagorical feture
    vec = DictVectorizer()  
    returnMat = vec.fit_transform(measure)

    '''
    #replcacec missing values into mean of values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    returnMat = imp.fit_transform(returnMat)
    return returnMat


TAGMAP = [dict() for i in range(40)]

def file2label(labelfile):
    label=[]
    fr = open(labelfile)
    lines = fr.readlines()  
    for line in lines:
        label.append(int(line.strip()))
    fr.close()
    return label
    

def main(trainfile, labelfile):
    mat = file2binMatrix(trainfile)
    label = np.array(file2label(labelfile))
    i, cri = 100, 'entropy'
    clf = RandomForestClassifier(n_estimators=i, criterion=cri)
    print "fitting...... %d %s" % (i, cri)
    roc_auc = cross_validation.cross_val_score(clf, mat, label,
                                               scoring='roc_auc', cv=2)
    print "Area under the ROC curve :", roc_auc

if __name__ == "__main__":
    main("data/orange_small_train.data", "data/orange_small_train_appetency.labels")
