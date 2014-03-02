CustomerPrediction
==================
What we have done and what we should do next

##Preprocessing:
* [ ] consider the missingness itself might predictive and added
* [x] only incode 10 most common values of each categorical attribute
* [ ] data clean (eliminate features either constant or duplicates)
* [ ] nomarlize to 0 ~ 1

##Classifiers:
* [x] random forests

   Area 0.67. Generally, entropy performs well than gini index and directly
  proportional to the number of estimators.
* [ ] boosted trees
* [ ] logistic regression
* [x] SGD classifier

##Evaluation:
* [ ] 2-fold cross-validataion approach
