import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame

df = pd.read_csv('./data/orange_small_train.data',sep='\t') # DataFrame
types = df.dtypes
cols = df.columns

from collections import Counter
from sklearn.feature_extraction import DictVectorizer

#drop columns with less than 100 non-missing values
for c in cols[types != 'object']:
    a = df[c]
    if len(a[a.notnull()]) < 100:
        df.drop(c,inplace=True,axis=1)

# types is float for [:190], object for [190:]
for c in cols[types == 'object']:
    labels = df[c]
    counter = Counter(labels[labels.notnull()])
    top_10 = [key for key,freq in counter.most_common(10) if freq > 100]
    
    d = [{c:key} for key in top_10]    
    X = [{c:key} for key in df[c]]
    
    v = DictVectorizer(sparse=False).fit(d)
    
    new_cols = v.transform(X)
    new_cols = DataFrame(new_cols)
    new_cols.columns = v.get_feature_names()

    for _ in new_cols.columns:
        df[_] = new_cols[_]
    df.drop(c,inplace=True,axis=1)

X = np.asarray(df)
y = np.asarray(map(int,[line.strip() for line in open('./data/orange_small_train_appetency.labels')]))

from sklearn.preprocessing import Imputer
X = Imputer(copy=False).fit_transform(X)

# begin training
from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

print cross_val_score(clf, X, y, scoring='roc_auc')
