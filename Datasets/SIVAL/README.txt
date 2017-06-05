AMIL-SIVAL

This is a modified version of the SIVAL repository for content-based image
retrieval. The original version of SIVAL can be obtained here:
http://www.cs.wustl.edu/accio/.

This version of the data has been enhanced with finer-granularity labels at
each image segment, or "instance" in the multiple instance (MI) setting. The
original repository contained 1500 images of 25 objects photographed in a
variety of locations and lighting conditions. Using a graphical interface, the
authors of the paper below manually labeled each segment as belonging to the
target object (positive:1) or not (negative:0) for 1499 of the images.
("sunI4_StripedNotebook_086"was sufficiently sun-bleached that the annotator
discarded it for our experiments).

The data are presented in C4.5 format similar to the UCI repository, with
".names" and ".data" files defining the feature set and values, respectively.
The 25 object tasks consist of 22 files each:

- XXX.names
- XXX.data
- XXX.[0-19].rep

Each ".rep" file contains the 20 random positive bags (images) and 20 random
negatives used to train the initial MI learning algorithm in the experiments
described in the citation below. The ".data" file contains bag (image) ID,
instance (segment) ID, and feature values for each instance, one per line,
comma delimited, and terminated by the class value (0,1) and a period.

FOR MORE INFORMATION:

B. Settles, M. Craven, & S. Ray (2008).
Multiple-Instance Active Learning.
Advances in Neural Information Processing Systems (NIPS) Volume 20.

http://pages.cs.wisc.edu/~bsettles/amil/