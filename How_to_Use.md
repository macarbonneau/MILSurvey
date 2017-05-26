# How to use #

This files explains how to use this code.


## MIL Data Format ##
Data sets are saved in an MILdataset object containing the following fields:

#### Basic Fields ####
X = all instance feature vectors
Y = instance labels inherited from of their bag
YR = the real instance label
B = the name of the bags
YB = is the labels of the bags
XtB = is the mapping from instance to bag (X to B)

#### Optional Fields for Special Cases ####
YP = predicted label by the algorithm
CX = code representing the instances (used with embedding methods)
CB = code representing the bags (used with embedding methods)
YS = is a vector used to tell if the instance were selected (usefull for certain algorithms).
QL = is a vector telling which instances have been queried in active learning.
QLB = is a vector telling which bags have been queried in active learning.
