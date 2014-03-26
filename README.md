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

   Area 0.72. 

* [x] boosted trees

    Area 0.75

* [x] SGD classifier

	Area 0.5

##Evaluation:
* [x] k-fold cross-validataion approach

   can replace k with any number, default is 3
