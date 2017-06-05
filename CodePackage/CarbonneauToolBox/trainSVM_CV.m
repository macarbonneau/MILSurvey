function [model, bestParameters, optStr] = trainSVM_CV(X,Y,opt)
%% This function train an SVM using LIBSVM with :
% X = data matrix. Each line is a data point and column are features.
% Y = A vector containing the label of each observation
%     (1=positive/0=negative)
% opt = a structure containing the configuration options for the SVM 
%       (C/gamma/kernel/etc.)  


%% Deal with class imbalance

imbalance = 1;
if isfield(opt,'classImbalance')
    if opt.classImbalance == 1
        % Compute imbalance
        % what is class 1
        c1 = Y(1);
        idx = Y ~= c1;
        c0 = Y(idx);
        c0 = c0(1);
        
        nc1 = sum(Y==c1);
        nc0 = length(Y)-nc1;
        
        if c1 == 1
            imbalance = nc1/nc0;
        else
            imbalance = nc0/nc1;
        end
        disp(['the imbalance in the data set is ' num2str(imbalance)])
    end
end

%% Create config
cfgT = createConfigTable(opt);

% check under which metric the best config is chosen
if strcmp(opt.metric,'AUC')
    AUC = true;
else
    AUC = false;
end

%% get performance for each configurations

estperf = zeros(size(cfgT,1),1);

if size(cfgT,1)>1
    parfor i = 1:size(cfgT,1)
        switch cfgT(i,1)
            case {0,1,2}
                optionStr = ['-s 0 -t ' num2str(cfgT(i,1)) ' -c ' num2str(cfgT(i,2))...
                    ' -w1 ' num2str(imbalance)  ' -d ' num2str(cfgT(i,4))...
                    ' -g ' num2str(cfgT(i,3)) ' -q'];
                estperf(i) = getPerformance(X,Y,optionStr,AUC);
        end
    end
    
    % chose the best config
    [val, idx] = max(estperf);
    disp(['The best config is ' num2str(cfgT(idx,:))])
    disp(['with ' num2str(val)])
    cfgT = cfgT(idx,:);
end


%% Re-order data set to avoid problem with score
idxp = Y==1;
X = [X(idxp,:);X(~idxp,:)];
Y = [Y(idxp,:);Y(~idxp,:)];

%% train model with the best parameter
switch cfgT(1,1)
    case {0,1,2}
        optStr = ['-s 0 -t ' num2str(cfgT(1,1)) ' -c ' num2str(cfgT(1,2))...
            ' -w1 ' num2str(imbalance)  ' -d ' num2str(cfgT(1,4))...
            ' -g ' num2str(cfgT(1,3)) ' -q'];
        model = svmtrain(Y,X,optStr);
    case {6}
        topt.C = cfgT(1,2);
        topt.kernel = 'eX2';
        topt.gamma = cfgT(1,3);
        model = svmKernelTrain(X,Y,topt);
end

bestParameters = cfgT;

end

function perf = getPerformance(X,Y,str,AUC)

nFolds = 5;
indexPerFold = listPerFold(Y,nFolds);

perf = zeros(nFolds,1);
for f = 1:nFolds
    
    [XT, YT, XV, YV] = getDatasetsForFold(X,Y,indexPerFold,f);
    
    model = svmtrain(YT,XT,str);
    [~, acc, DV] = svmpredict(YV,XV, model);
    perf(f) = acc(1);
    if AUC
        if YT(1) < 1
            DV = -DV;
        end
        
        perf(f) = getAUROC(DV, YV, 0.1);
    end
    
end

perf = mean(perf);

end

function perf = getPerformanceBow(X,Y,cfg,AUC)

nFolds = 5;
indexPerFold = listPerFold(Y,nFolds);

opt.C = cfg(2);
opt.kernel = cfg(1);
opt.gamma = cfg(3);

perf = zeros(nFolds,1);
for f = 1:nFolds
    
    [XT, YT, XV, YV] = getDatasetsForFold(X,Y,indexPerFold,f);
    
    model = svmKernelTrain(XT,YT,opt);
    
    % Test the model
    [PL, acc, DV] = svmKernelTest(XV,YV,opt,model);
    
    % save performance
    perf(f) = acc(1);
    if AUC
        DV = correctDV(DV,PL);
        perf(f) = getAUROC(DV, YV, 0.1);
    end
    
end

perf = mean(perf);

end

function [XTR, YTR, XVAL, YVAL] = getDatasetsForFold(X,Y,indexPerFold,valFoldNumber)
% Cette fonction retourne deux prdataset un pour la validation (VAL)
% et l'autre pour l'entrainement TR

% X : toutes les données dans une  matrice
% Y : les labels correspondant à X
% indexPerFold : est une structure de liste d'index obtenue avec la
% fonction obtenirListeParFold
% valFoldNumber: Est le numéro du fold utilisé pour la validation


idxTR = [];

for f = 1:size(indexPerFold,1)
    
    if f == valFoldNumber
        idxVal = indexPerFold{f};
    else
        idxTR = [idxTR;indexPerFold{f}];
    end
end

XTR = X(idxTR,:);
XVAL = X(idxVal,:);

YTR = Y(idxTR,:);
YVAL = Y(idxVal,:);

end

function [indexPerFold] = listPerFold(Y,nFolds)

% X est un prdataset
% nFolds est le nombre de folds désiré
% La fonction retourne une structure contenant des listes
% d'index. Il y a une liste d'index par fold.
indexPerFold = cell(nFolds,1);

idx = 1:size(Y,1);

% obtenir la liste de toutes les classes
Ynames = unique(Y);

ctr = 0;
for c = 1:length(Ynames)
    
    tmpI = idx(Y==Ynames(c));
    rp = randperm(length(tmpI));
    tmpI = tmpI(rp);
    
    for i = 1:length(tmpI)
        f = mod(ctr,nFolds)+1;
        indexPerFold{f} = [indexPerFold{f};tmpI(i)];
        ctr = ctr+1;
    end
    
end

end

function [DV] = correctDV(DV,PL)
sp = sum(DV(PL==1));
sn = sum(DV(PL~=1));

if sp < sn
    DV = -DV;
end

end


function cfgT  = createConfigTable(opt)

cfgT = [];   % [kernel C gamma degree]

for i = opt.kernel
    
    if i == 0
        for c = opt.C
            tmp = [0 c 0 0];
            cfgT = [cfgT; tmp];
        end
        
    elseif i == 2
        for c = opt.C
            for g = opt.gamma
                tmp = [2 c g 0];
                cfgT = [cfgT; tmp];
            end
        end
        
    elseif i == 1
        for c = opt.C
            for d = opt.degree
                tmp = [1 c 0 d];
                cfgT = [cfgT; tmp];
            end
        end
        
    elseif i == 5
        for c = opt.C
            tmp = [5 c 0 0];
            cfgT = [cfgT; tmp];
        end

        
    elseif i == 6
        for c = opt.C
            for g = opt.gamma
                tmp = [6 c g 0];
                cfgT = [cfgT; tmp];
            end
        end
    else
        error(['The kernel ' num2str(i) ' is not supported'])
    end
    
end

end
