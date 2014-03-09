#COMP4432 Assignment 1 report

##Data exploration
In this project, we only focus on the fast track challenge. In the fast
track challenge, each instance contains 230 variables during which 190 of
them are numerical variables and 40 of them are categorical variables.
Therefore, we need to combine numerical variables and categorical variables
together in order to put them into classification. One method is employ
binarization to categorical variables. But this method will result in very
high dimension sparse matrix and thus decrease computation efficiency.
Another method is employ ordinalization, which means using different
numbers to represent different category values for each attribute. The
drawback of this method is that arithmetic operations cannot be applied to
categorical attributes even when the coding of their values is expressed by
integer numbers. Therefore, those classification methods which involves
arithmetic operations on attribute values may not perform well when using
this method. Both above methods are tried in our project to maximum
predict accuracy, then we find the binarization method can achieve higher
ROC value while consuming more time training.

We also noticed that many variables have a large number of missing values
and some of the categorical have a huge vocabulary. For the missing
numerical values, we followed the standard approach of imputing the missing
value by the mean of the features. And for the missing categorical values,
we treat them as a separate value.


##Classification methods
