function out =  NSKSVM(D,operation,model)


switch operation
    case 'train'
        
        % get best configuration
        [C, gamma] = validationMIkernel(D);
        
        % convert data set to the pr format
        D = convertDatasetForMILToolbox(D);
        
        % get the model
        out = trainMIkernelSVM(D,gamma,C);
        
    case 'test'
        
        % give true labels
        out.TL = D.YR;
        out.PL = out.TL;
        out.SC = out.TL;
        out.TLB = D.YB;
        % convert data set to the pr format
        D = convertDatasetForMILToolbox(D);
        
        % classify instances and bags
        [out.PLB,out.SCB,out.TLB] = testMIkernelSVM(D,model);
end

end


function [C, gamma] = validationMIkernel(D)

% parameters
C = [0.01 0.1 1 10 100];
gamma = [0.125 0.25 0.5 1 2 4 8 16 32];

nFolds = 8;
tic
% matrix for saving results
AUC = zeros(length(C),length(gamma),nFolds);

BagPerFoldList = divideBagsInFolds(nFolds,D);

for f = 1:nFolds
    
    statusBar(f,nFolds)
    [TRD, TED] = getTrainingAndTestDatasets(f,nFolds,BagPerFoldList,D);
    
    % convert data set
    disp(['Data set Conversion fold ' num2str(f)])
    tic
    TRD = convertDatasetForMILToolbox(TRD);
    TED = convertDatasetForMILToolbox(TED);
    disp(['Done'])
    toc
    
    parfor c = 1:length(C)
        
        AUCtmp = zeros(1,length(gamma));
        
        for g = 1:length(gamma)
            
            % train a model
            model = trainMIkernelSVM(TRD,gamma(g),C(c));
            
            % test the model
            [PL,SC,TL] = testMIkernelSVM(TED,model);
            
            perf = getClassifierPerfomance(PL,TL,SC);
            
            AUCtmp(g) = perf.AUC;
            
            disp(['C = ' num2str(C(c)) ' gamma = ' num2str(gamma(g))...
                '  AUC = ' num2str(AUCtmp(g))]);
            
        end
        
        fn = ['temp/tmpValmik' num2str(C(c))];
        dlmwrite(fn,AUCtmp);
        
    end
    
    for c = 1:length(C)
        fn = ['temp/tmpValmik' num2str(C(c))];
        tmp = dlmread(fn);
        AUC(c,:,f) = tmp;
        
    end
end

AUC = mean(AUC,3);

% get the best performing parameters
[~,ind] = max(AUC(:));
[c,g] = ind2sub(size(AUC),ind);

disp(['the best config for MI-kernel is : C=' num2str(C(c)) ' and g=' num2str(gamma(g))])

C = C(c);
gamma = gamma(g);
toc
end


function model = trainMIkernelSVM(D,gamma,C)

tmp = cell(1);
tmp{1} = gamma;

W = milproxm(D,'miRBF',tmp);
[K,Y] = getKernel(W,D);
optionStr = ['-s 0 -t 4 -c ' num2str(C) ' -q'];
model.SVM = svmtrain(Y,[(1:size(K,1))' K],optionStr);
model.kernel = W;

end

function [K,Y] = getKernel(kernelmap,D)

[~,nlab,bagid] = getbags(D);
nlab = ispositive(nlab);

if ~istrained(kernelmap)
    trkernelmap = D*kernelmap;
else
    trkernelmap = kernelmap;
end
n = size(bagid,1);
% compute bag kernel:
K = +(D*trkernelmap);

i = -30;
while (pd_check(K + (10.0^i)*eye(n)) == 0)
    i = i + 1;
end
if (i > -30),
    prwarning(2,'K is not positive definite. The kernel is regularized by adding 10.0^(%d)*I',i);
end
K = K + (10.0^(i+2))*eye(n);

% Compute the parameters for the optimization:
Y = 2*nlab-1;

end

function [PL,SC,TL] = testMIkernelSVM(D,model)

[~,nlab] = getbags(D);
nlab = ispositive(nlab);
Y = 2*nlab-1;

% compute bag kernel:
K = +(D*model.kernel);

% classify with the SVM
[~, ~, SC] = svmpredict(Y, [(1:size(K,1))' K], model.SVM);
SC = SC+model.SVM.rho;

PL = SC>0;
[~,nlab,~,~] = getbags(D);
TL = ispositive(nlab);

end

