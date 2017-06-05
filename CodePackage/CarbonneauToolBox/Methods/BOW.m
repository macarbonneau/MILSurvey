function out =  BOW(D,DT,operation,model,opt)
%% This function implements a Bag-of-word method where the dictionary is
% learned with k-means and the classifier is an SVM with an exponential
% Chi-square kernel.
% D is a MILdataset object for the training data
% DT is a MILdataset object for the test data


switch operation
    case 'train'
        
        % get best configuration
        [opt.DS] = validationBow(D,opt);
        
        % get the model
        [out] = trainBow(D,opt);
        
    case 'test'
        
        % give true labels
        out.TL = DT.YR;
        out.PL = out.TL;
        out.SC = out.TL;
        out.TLB = DT.YB;
        
        [out.PLB,out.SCB] = testBoW(D,DT,model);    
end

end


function [DS] = validationBow(D,opt)

nFolds = 8;
tic
% matrix for saving results
AUC = zeros(length(opt.DS),nFolds);

BagPerFoldList = divideBagsInFolds(nFolds,D);

for f = 1:nFolds
    
    statusBar(f,nFolds)
    [TRD, TED] = getTrainingAndTestDatasets(f,nFolds,BagPerFoldList,D);
    
    for d = 1:length(opt.DS)
        
        
        % learn dictionary
        [dict, A] = vl_kmeans(TRD.X', opt.DS(d),'NumRepetitions',20,'Initialization','plusplus');
        dict = dict';
        TRD.CX = double(A)';
        
        % encode test data
        TED.CX = wordAssociator(TED.X,dict);
        
        % Create an histogram
        TRD = createHistograms(TRD,opt.DS(d));
        TED = createHistograms(TED,opt.DS(d));
        
        AUCgamma = zeros(1,length(opt.gamma));
        for g = 1:length(opt.gamma)
            
            % Compute chi2 kernel
            [K, KT] = chi2Kernel(TRD,TED,opt.gamma(g));
            
            % train clasifier
            [model] = trainSVM_CV(K,TRD.YB,opt);
            % test it
            [PL, acc, SC] = svmpredict(TED.YB,KT, model);
            % compute perf
            TL = TED.YB;
            perf = getClassifierPerfomance(PL,TL,SC);
            AUCgamma(g) = perf.AUC;
            
        end
        
        AUC(d,f) = max(AUCgamma);
        disp(['DS = ' num2str(opt.DS(d)) '  AUC = ' num2str(AUC(d,f))]);
        
    end
    
    
end

AUC = mean(AUC,2);

% get the best performing parameters
[~,ind] = max(AUC(:));
[d] = ind2sub(size(AUC),ind);

disp(['the best config for BoW is : DS=' num2str(opt.DS(d))])
DS = opt.DS(d);
end

function D = createHistograms(D,nW)

D.CB = zeros(length(D.B),nW);

for b = 1:length(D.B)
    
    D.CB(b,:) = hist(D.CX(D.XtB==D.B(b)),nW);
    
end

%if opt.histogramNormalization
% denom = repmat(sum(D.CB,2),1,nW);
% D.CB = D.CB./denom;
%end

end

function [model] = trainBow(D,opt)

% learn dictionary
[dict, A] = vl_kmeans(D.X', opt.DS,'NumRepetitions',20,'Initialization','plusplus');
dict = dict';
D.CX = double(A)';

% Create an histogram
D = createHistograms(D,opt.DS);

nFolds = 5;
BagPerFoldList = divideBagsInFolds(nFolds,D);
AUCgamma = zeros(nFolds,length(opt.gamma));

for f = 1:nFolds
    
    [TRD, TED] = getTrainingAndTestDatasets(f,nFolds,BagPerFoldList,D); 
    
    for g = 1:length(opt.gamma)
        
        % Compute chi2 kernel
        [K, KT] = chi2Kernel(TRD,TED,opt.gamma(g));
        
        % train clasifier
        model = trainSVM_CV(K,TRD.YB,opt);
        % test it
        [PL, acc, SC] = svmpredict(TED.YB,KT, model);
        % compute perf
        TL = TED.YB;
        perf = getClassifierPerfomance(PL,TL,SC);
        AUCgamma(f,g) = perf.AUC;
    end
end

[AUC, idx] = max(mean(AUCgamma));
opt.gamma = opt.gamma(idx(1));

% train clasifier
[K] = chi2Kernel(D,[],opt.gamma);
[model.SVM] = trainSVM_CV(K,D.YB,opt);

model.dict = dict;
model.gamma = opt.gamma;

end

function [PL,SC] = testBoW(D,DT,model)

DS = size(model.dict,1);

% encode test data
D.CX = wordAssociator(D.X,model.dict);
DT.CX = wordAssociator(DT.X,model.dict);

        
% Create an histogram
D = createHistograms(D,DS);
DT = createHistograms(DT,DS);

% Test the model
[~, KT] = chi2Kernel(D,DT,model.gamma);
[PL, acc, SC] = svmpredict(DT.YB,KT, model.SVM);

end

function [K, KT] = chi2Kernel(TRD,TED,gamma)


%% Encode all bags in the training set
NTR = size(TRD.B,1);
K = zeros(NTR);
for i = 1:NTR
    for j = 1:NTR
        
        xe = TRD.CB(i,:);
        xr = TRD.CB(j,:);
        
        K(i,j) = exp(-gamma*sum((xr-xe).^2./(xr+xe+eps)));
    end
end


%% Encode all bags in the test set
if ~isempty(TED)
    
    NTE = size(TED.B,1);
    KT = zeros(NTE,NTR);
    
    for i = 1:NTE
        for j = 1:NTR
            
            xe = TED.CB(i,:);
            xr = TRD.CB(j,:);
            
            KT(i,j) = exp(-gamma*sum((xr-xe).^2./(xr+xe+eps)));
        end
    end
    
end
end

function code = wordAssociator(word,dict)

code = zeros(size(word,1),1);
D = zeros(size(dict,1),1);
for i = 1:size(word,1)
    
for j = 1:size(dict,1)
    
    D(j) = sum((word(i,:)-dict(j,:)).^2);    
end
    
[~,code(i)] = min(D);

end

end
