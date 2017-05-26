# MILSurvey

This is the Matlab code used for the experiments in the paper:
[1] M.-A. Carbonneau, V. Cheplygina, E. Granger, and G. Gagnon, “Multiple Instance Learning: A Survey of Problem Characteristics and Applications,” ArXiv e-prints, vol. abs/1612.0, 2016. 


This code has many dependencies on various toolboxes:

1) The MIL Toolbox
This is where many of the algorithm implementation come from.
http://prlab.tudelft.nl/david-tax/mil.html
[2] C. V Tax D.M.J., “{MIL}, A {M}atlab Toolbox for Multiple Instance Learning.” Jun-2016.

2) The PRTools
This is necessary to run the MIL Toolbox.
http://prtools.org/

3) Dd_tools
This is necessary to run the MIL Toolbox.
http://prlab.tudelft.nl/david-tax/dd_tools.html
[3] D. M. J. Tax, “DDtools, the Data Description Toolbox for Matlab.” Jun-2015.

4) LIBSVM
This is the implementation used for all SVM in the experiments.
https://www.csie.ntu.edu.tw/~cjlin/libsvm/
[4] C.-C. Chang and C.-J. Lin, “LIBSVM: A Library for Support Vector Machines,” ACM Trans. Intell. Syst. Technol., vol. 2, no. 3, May 2011.

5) SLEP
This is for sparse learning but I don't remember what it is used for...   might be legacy from previous experiments. sorry...
http://yelab.net/software/SLEP/
[5] J. Liu, S. Ji, and J. Ye. SLEP: Sparse Learning with Efficient Projections. Arizona State University, 2009. 

6) EMD
A package for earth mover's distance. A do not remember where it is from but it is included in the code here. sorry again.... 

7) VLFeat 
This is used only for the implementation of k-means in RSIS. It can be replaced by any other implementation if necessary.
http://www.vlfeat.org/
[6] A. Vedaldi and B. Fulkerson, “{VLFeat}: An Open and Portable Library of Computer Vision Algorithms.” 2008.
