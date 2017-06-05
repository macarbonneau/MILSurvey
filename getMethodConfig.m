function opt = getMethodConfig(method,dataset,expType)
% INPUTS:
% method is a string containing the name of the MIL methods to use
% dataset is a string containing the name of the data set
% expType is a string containing the ype of experiment to conduct
% OUTPUT:
% opt is an object containing the field used to configure the method
% specified for the dataset and experiment type.


opt.method = method;
opt.dataset = dataset;
opt.expType = expType;


switch method
    
    case 'EMDD'
        opt = cfgEMDD(opt)
        
    case 'MILES'
        
    case 'RSIS'
        opt = cfgRSIS(opt)
        
    case {'kNN','SIkNN'}
        
    case 'MILBoost'
        opt.boostingRounds = 100;
        
    case {'CKNN','CkNN','C-KNN','C-kNN'}
            
    case 'CCE'
        
    case 'miSVM'
        opt = cfgmiSVM(opt)
        
    case {'SISVM','SI-SVM'}
        opt = cfgSISVM(opt)
        
    case 'MI-SVM'
        opt = cfgMMISVM(opt)
        
    case {'NSK-SVM','MI-kernel'}
        
    case 'EMD-kernel'
        opt.C = [0.1 1 10 100]; % for SVM
        opt.gamma = [0.001 0.01 0.1 1 10]; % SVM gaussian kernel
        opt.kernel = [0 2];
        opt.metric = 'Acc';
        
    case {'migraph','miGraph','mi-graph'}
        opt.method = 'migraph';
    case 'BoW'
        opt = cfgBOW(opt)
        
    case {'SI-SVM-TH','SISVMTH'}
        opt = cfgSISVMTH(opt)
        
    case {'MInD'}    
        opt.C = [0.1 1 10 100]; % for SVM
        opt.gamma = [0.001 0.01 0.1 1 10]; % SVM gaussian kernel
        opt.kernel = [0 2];
        opt.metric = 'Acc';
        
    otherwise
        error('UNKNOWN METHOD')
end

end


function opt = cfgRSIS(opt)

if strfind(lower(opt.dataset),'news')
    optName = 'Newsgroups';
else
    optName = opt.dataset;
end

switch optName
    
    case 'Letters'
       % grid search parameters
        opt.C = [10 100]; % for SVM
        opt.gamma = [0.00001 0.0001 0.001 0.01 0.1]; % SVM gaussian kernel
        opt.kernel = [2];
        opt.metric = 'AUC'; % metric used in validation
        opt.NDSS = [0.5]; % proportion of dimension in random subspaces
        opt.nK = [20]; % number of clusters in k-means
        opt.T = [0.01]; % temperature for soft max selection
        opt.nRSS = 1000; % number of generated subspaces
        opt.ES = 50; % ensemble size
    case {'Newsgroups','Spam'}
        % grid search parameters
        opt.C = [10 100]; % for SVM
        opt.gamma = [0.5 1]; % SVM gaussian kernel
        opt.kernel = [2];
        opt.metric = 'AUC'; % metric used in validation
        opt.NDSS = [0.05]; % proportion of dimension in random subspaces
        opt.nK = [5]; % number of clusters in k-means
        opt.T = [0.01]; % temperature for soft max selection
        opt.nRSS = 1000; % number of generated subspaces
        opt.ES = 50; % ensemble size
        
    case 'Particles'
        % grid search parameters
        opt.C = [0.1 1 10 100]; % for SVM
        opt.gamma = [0.01 0.1 1 10]; % SVM gaussian kernel
        opt.kernel = [2];
        opt.metric = 'AUC'; % metric used in validation
        opt.NDSS = [0.1 0.2 0.3]; % proportion of dimension in random subspaces
        opt.nK = [5 10 20]; % number of clusters in k-means
        opt.T = [0.01, 0.001]; % temperature for soft max selection
        opt.nRSS = 1000; % number of generated subspaces
        opt.ES = 100; % ensemble size
    case {'ToyUD','ToyCD'}
        % grid search parameters
        opt.C = [0.1 1 10 100]; % for SVM
        opt.gamma = [0.01 0.1 1 10]; % SVM gaussian kernel
        opt.kernel = [2];
        opt.metric = 'AUC'; % metric used in validation
        opt.NDSS = [0.1 0.2 0.3]; % proportion of dimension in random subspaces
        opt.nK = [5 10 20]; % number of clusters in k-means
        opt.T = [0.01, 0.001]; % temperature for soft max selection
        opt.nRSS = 1000; % number of generated subspaces
        opt.ES = 100; % ensemble size
    case 'Birds'
        % grid search parameters
        opt.C = [0.1 1 10 100]; % for SVM
        opt.gamma = [0.01 0.1 1 10]; % SVM gaussian kernel
        opt.kernel = [2];
        opt.metric = 'AUC'; % metric used in validation
        opt.NDSS = [0.1 0.2 0.3]; % proportion of dimension in random subspaces
        opt.nK = [5 10 20]; % number of clusters in k-means
        opt.T = [0.01, 0.001]; % temperature for soft max selection
        opt.nRSS = 1000; % number of generated subspaces
        opt.ES = 100; % ensemble size
    case 'SIVAL'
        opt.C = [10]; % for SVM
        opt.gamma = [0.00001 0.0001 0.001 0.01 0.1]; % SVM gaussian kernel
        opt.kernel = [2];
        opt.metric = 'AUC'; % metric used in validation
        opt.NDSS = [0.1]; % proportion of dimension in random subspaces
        opt.nK = [5]; % number of clusters in k-means
        opt.T = [0.1]; % temperature for soft max selection
        opt.nRSS = 1000; % number of generated subspaces
        opt.ES = 50; % ensemble size
    otherwise
        disp(['Dont know how to configure for ' opt.dataset])
        opt.C = [10]; % for SVM
        opt.gamma = [0.0001 0.001 0.01 0.1]; % SVM gaussian kernel
        opt.kernel = [2];
        opt.metric = 'AUC'; % metric used in validation
        opt.NDSS = [0.05]; % proportion of dimension in random subspaces
        opt.nK = [5]; % number of clusters in k-means
        opt.T = [0.01]; % temperature for soft max selection
        opt.nRSS = 500; % number of generated subspaces
        opt.ES = 100; % ensemble size
end

end

function opt = cfgEMDD(opt)

switch opt.dataset
    
    case 'Letters'
        opt.epochs = [3 3];
        opt.tol =  [1e-4 1e-3 1e-4 1e-4];
        opt.NP = 48; % number of prototype
        
    case {'Newsgroups','Spam'}
        opt.epochs = [3 3];
        opt.tol =  [1e-3 1e-3 5e-4 5e-4];
        opt.NP = 48; % number of prototype
        
    case 'Particles'
        opt.epochs = [3 3];
        opt.tol =  [1e-4 1e-4 1e-4 1e-4];
        opt.NP = 48; % number of prototype
    case 'Birds'
        opt.epochs = [3 3];
        opt.tol =  [1e-4 1e-4 1e-4 1e-4];
        opt.NP = 48; % number of prototype
    case  {'ToyUD','ToyCD'}
        opt.epochs = [3 3];
        opt.tol =  [1e-4 1e-3 1e-4 1e-4];
        opt.NP = 24; % number of prototype
    otherwise
        disp(['Dont know how to configure for ' opt.dataset])
        opt.epochs = [3 3];
        opt.tol =  [1e-4 1e-3 1e-4 1e-4];
        opt.NP = 48; % number of prototype
end

end

function opt = cfgmiSVM(opt)

switch opt.dataset
    
    case 'Letters'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.nIter = 30;
    case {'Newsgroups','Spam'}
        opt.C = [0.1 1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.nIter = 30;
    case 'Particles'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.nIter = 30;
    case 'Birds'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.nIter = 30;
    case  {'ToyUD','ToyCD'}
        opt.C = [1 10 100 1000];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.nIter = 30;
    otherwise
        disp(['Dont know how to configure for ' opt.dataset])
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.nIter = 30;
end

end

function opt = cfgMMISVM(opt)

switch opt.dataset
    
    case 'Letters'
        opt.C = [1 10 100];
        opt.gamma = [0.01 0.1 1 10 100 1000];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.maxIter =  100;
    case {'Newsgroups','Spam'}
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.maxIter =  100;
    case 'Particles'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.maxIter =  100;
    case 'Birds'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.maxIter =  100;
    case  {'ToyUD','ToyCD'}
        opt.C = [1 10 100 1000];
        opt.gamma = [0.01 0.1 1 10 100];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.maxIter =  100;
    otherwise
        disp(['Dont know how to configure for ' opt.dataset])
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        opt.maxIter =  100;
end

end

function opt = cfgSISVMTH(opt)

switch opt.dataset
    
    case 'Letters'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
    case {'Newsgroups','Spam'}
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
    case 'Particles'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
    case 'Birds'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
    case  {'ToyUD','ToyCD'}
        opt.C = [1 10 100];
        opt.gamma = [0.01 0.1 1 10 ];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
    otherwise
        disp(['Dont know how to configure for ' opt.dataset])
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
end

end

function opt = cfgSISVM(opt)

switch opt.dataset
    
    case 'Letters'
        opt.C = [1 10 100 ];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        
    case {'Newsgroups','Spam'}
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        
    case 'Particles'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
    case 'Birds'
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        
    case  {'ToyUD','ToyCD'}
        opt.C = [1 10 100];
        opt.gamma = [0.01 0.1 1 10 100];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
        
    otherwise
        disp(['Dont know how to configure for ' opt.dataset])
        opt.C = [1 10 100 ];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [2];
        opt.degree = [2];
        opt.metric = 'AUC';
end

end

function opt = cfgBOW(opt)

switch opt.dataset
    
    case 'Letters'
        opt.DS = [5 10 15 20 30 50 100 200];   % dictionary size
        opt.C = [0.001 1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [0];
        opt.metric = 'AUC';
    case {'Newsgroups','Spam'}
        opt.DS = [5 10 15 20 30 50];   % dictionary size
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = 0;
        opt.metric = 'AUC';
    case 'Particles'
        opt.DS = [5 10 15 20 30 50];   % dictionary size
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = 0;
        opt.metric = 'AUC';
    case 'Birds'
        opt.DS = [5 10 15 20 30 50];   % dictionary size
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = 0;
        opt.metric = 'AUC';
    case  {'ToyUD','ToyCD'}
        opt.DS = [5 10 15 20 30 50];   % dictionary size
        opt.C = [1 10 100 1000 10000];
        opt.gamma = [0.01 0.1 1 10 100 1000];
        opt.kernel = 0;
        opt.metric = 'AUC';
    case 'SIVAL'
        opt.DS = [15 20 30 50 100 200 400];   % dictionary size
        opt.C = [0.001 1 10 100];
        opt.gamma = [0.001 0.01 0.1 1 10];
        opt.kernel = [0];
        opt.metric = 'AUC';
    otherwise
        disp(['Dont know how to configure for ' opt.dataset])
        opt.DS = [10 50];   % dictionary size
        opt.C = [1 10 100];
        opt.gamma = [0.001 0.01 0.1 1];
        opt.kernel = [0];
        opt.metric = 'AUC';
end

end

