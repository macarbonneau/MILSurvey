function [] = mainTestFunction(allMethods,dataset)

close all
clc
%% LOAD PACKAGES
addpath(genpath('CodePackage/CarbonneauToolBox'))
addpath('CodePackage/LIBSVM/libsvm-3.20/matlab')
run('CodePackage/vlfeat/toolbox/vl_setup')
addpath(genpath('CodePackage/prtools'))
addpath(genpath('CodePackage/milToolbox'))
addpath(genpath('CodePackage/dd_tools'))
addpath(genpath('CodePackage/EMD'))

%% PARAMETERS

if ischar(allMethods)
    allMethods = cellstr(allMethods);
end
[dataset,fn] = getParForDataset(dataset);

%% LOAD DATASET

load(fn);
disp('=============================================================')
disp(['= DATA SET ACQUIRED: ' fn])
disp('-------------------------------------------------------------')

%% TEST METHODS ON THE DATA SET

for i = 1:length(allMethods)
    tic
    method = allMethods{i};
    disp(['= Method: ' method])
    
    % get config for test
    opt = getMethodConfig(method,dataset,'single');
        
    if exist('DT','var')
        % perform normalization if necessary
        [D,DT] = normalizeDataSet(D,DT,opt);
        [perf,perfB] = performExperimentWithTestSet(D,DT,method,opt);
    else
        % perform normalization if necessary
        [D] = normalizeDataSet(D,[],opt);
        [perf,perfB] = performExperimentWithCrossVal(D,method,opt);
    end
    
    
    %% Results
    disp('=============================================================')
    disp(['= ' method])
    disp('-------------------------------------------------------------')
    
    disp('- instances')
    disp(['AUC: ' num2str(perf.AUC) ])
    disp(['UAR: ' num2str(perf.UAR) ])
    
    disp('- bags')
    disp(['AUC: ' num2str(perfB.AUC) ])
    disp(['UAR: ' num2str(perfB.UAR) ])
    toc
    disp('-------------------------------------------------------------')
    
    % save results
    fn = ['Results/' dataset '-' method '-' date];
    save(fn,'perf','perfB');
    
end


end


function [perf,perfB] = performExperimentWithCrossVal(D,method,opt)

nRep = 10;
nFolds = 10;
perfObj = cell(nRep,nFolds);
perfObjB = cell(nRep,nFolds);

for r = 1:nRep
    BagPerFoldList = divideBagsInFolds(nFolds,D);
    for fold = 1:nFolds
        disp(['---- Performing Fold ' num2str(fold) ' of rep ' num2str(r)]);tic
        
        % create training and test datasets
        [TRD, TED] = getTrainingAndTestDatasets(fold,nFolds,BagPerFoldList,D);
        % train and test for this fold
        [pred] = trainAndTestMIL(TRD, TED, method, opt);
        % get performance
        perfObj{r,fold} = getClassifierPerfomance(pred.PL,pred.TL,pred.SC);
        perfObjB{r,fold} = getClassifierPerfomance(pred.PLB,pred.TLB,pred.SCB);toc
    end
end

perf  = getMeanPref(perfObj);
perfB = getMeanPref(perfObjB);

end

function [perf,perfB] = performExperimentWithTestSet(D, DT, method,opt)

[pred] = trainAndTestMIL(D, DT, method, opt);

%% Compute Performances
[perf] = getClassifierPerfomance(pred.PL,pred.TL,pred.SC);
[perfB] = getClassifierPerfomance(pred.PLB,pred.TLB,pred.SCB);


end

function meanPerf = getMeanPref(perfO)

fName = fieldnames(perfO{1,1});

for i = 1:length(fName)
    meanPerf.(fName{i}) = [0 0];
end

for i = 1:length(fName)
    table = zeros(size(perfO));
    for j = 1:size(perfO,1)
        for k = 1:size(perfO,2)
            table(j,k) = perfO{j,k}.(fName{i});   
        end
    end
    
    m =  mean(table,2);
    meanPerf.(fName{i}) = [mean(table(:)) std(m)];
    
end

end

function [dataset,fn] = getParForDataset(dataset)

switch lower(dataset)
    
    case {'musk1'}       
        fn = ['Datasets/Musk/Musk1'];
        
    case {'musk2'}        
        fn = ['Datasets/Musk/Musk2'];
        
    case {'tiger','fox','elephant'}        
        fn = ['Datasets/FoxElephantTiger/' dataset];
        
    case {'test'}
        fn = ['Datasets/Test/test'];
    
    case {'testcv'}
        fn = ['Datasets/Test/testCV'];
        
end
end

function [D,DT] = normalizeDataSet(D,DT,opt)

if isfield(opt,'dataNormalization')
    switch opt.dataNormalization
        
        case {'std'}
            % X is a dataset with row entries
            u = mean([D.X ; DT.X]);
            s = std([D.X ; DT.X])+eps;
            um = repmat(u,size(D.X,1),1);
            sm = repmat(s,size(D.X,1),1);
            D.X = (D.X-um)./sm;
            um = repmat(u,size(DT.X,1),1);
            sm = repmat(s,size(DT.X,1),1);
            DT.X = (DT.X-um)./sm;
            
        case {'var','variance'}
            % X is a dataset with row entries
            u = mean([D.X ; DT.X]);
            s = var([D.X ; DT.X])+eps;
            um = repmat(u,size(D.X,1),1);
            sm = repmat(s,size(D.X,1),1);
            D.X = (D.X-um)./sm;
            um = repmat(u,size(DT.X,1),1);
            sm = repmat(s,size(DT.X,1),1);
            DT.X = (DT.X-um)./sm;
            
        case {'0-1'}
            ma = max([D.X ; DT.X]);
            mi = min([D.X ; DT.X]);
            mam = repmat(ma,size(D.X,1),1);
            mim = repmat(mi,size(D.X,1),1);
            D.X = (D.X-mim)./(mam-mim);
            mam = repmat(ma,size(DT.X,1),1);
            mim = repmat(mi,size(DT.X,1),1);
            DT.X = (DT.X-mim)./(mam-mim);
            
    end 
end

end
