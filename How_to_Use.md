# How to use #

This files explains how to use this code. Make sure you installed all dependencies.


## MIL Data Format ##
By convention data set are called D. If there is a test data set it is called DT.
Data sets are saved in an MILdataset object containing the following fields:

#### Basic Fields ####
* X = all instance feature vectors
* Y = instance labels inherited from of their bag
* YR = the real instance label
* B = the name of the bags
* YB = is the labels of the bags
* XtB = is the mapping from instance to bag (X to B)

#### Optional Fields for Special Cases ####
* YP = predicted label by the algorithm
* CX = code representing the instances (used with embedding methods)
* CB = code representing the bags (used with embedding methods)
* YS = is a vector used to tell if the instance were selected (usefull for certain algorithms).
* QL = is a vector telling which instances have been queried in active learning.
* QLB = is a vector telling which bags have been queried in active learning.

## Performing an experiment ##
1) Use the function called mainTestFucntion.m
2) Specify which methods you want to use.
3) Specify the name of the data set.

For example, if you want to test mi-SVM and MInD on the Musk1 data set, write :
mainTestFunction({'miSVM','MInD'},'musk1')

When the data set is contained in a single object called D, the experiment will be performed as a 10x10-fold cross-validation. If a test set (DT) is included, the methods are trained on D and the results obtained on DT are reported. 
