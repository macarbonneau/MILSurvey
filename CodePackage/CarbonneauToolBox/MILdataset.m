classdef MILdataset
    % MILDATASET this is an object containing al information for a MIL
    % dataset
    %   X = is all instance feature vectors
    %   Y = is the label of the bag for each instance
    %   B = is the name of the bags
    %   YB = is the labels of the bags
    %   XtB = is the mapping from instance to bag (X to B)
    %   YR = is the real instance label (optional)
    %   YP = predicted label by the algorithm
    %   CX = code representing the instances (used with embedding methods)
    %   CB = code representing the bags (used with embedding methods)
    %   YS = is a vector used to tell if the instance are selected to
    %        represent their bag.
    %   QL = is a vector telling which instances have been queried in active
    %        learning.
    %   QLB = is a vector telling which bags have been queried in active
    %        learning.
    
    properties
        
        X = [];
        Y = [];
        B = [];
        YB = [];
        XtB = [];
        YR = [];
        YP = [];
        CX = [];
        CB = [];
        YS = [];
        QL = [];
        QLB = [];
    end
    
    
end

