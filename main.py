#!/opt/local/bin/python
import numpy as np
import pylab as pl
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn.metrics import roc_curve, auc

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

def file2label(labelfile):
    label=[]
    fr = open(labelfile)
    lines = fr.readlines()  
    for line in lines:
        label.append(int(line.strip()))
    return label
    

def main(trainfile, labelfile):
    mat = file2matrix(trainfile)
    label = file2label(labelfile)
    dataSize = 100
    X_train, X_test = mat[:dataSize], mat[dataSize:]
    y_train, y_test = label[:dataSize], label[dataSize:]
    classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
    probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()

if __name__ == "__main__":
    main("data/orange_small_test.data", "data/orange_small_train_appetency.labels")
