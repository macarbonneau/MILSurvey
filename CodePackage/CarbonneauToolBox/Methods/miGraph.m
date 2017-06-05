
function out =  miGraph(D,operation,model)


switch operation
    case 'train'
        
        % get best configuration
        [distPar, C, gamma] = validationMIkernel(D);
        
        % convert data set to the pr format
        D = convertDatasetForMILToolbox(D);
        
        % get the model
        opt = cell(1,2);
        opt{1} = distPar;
        opt{2} = gamma;
        out = trainMIkernelSVM(D,opt,C);
        
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


function [distPar, C, gamma] = validationMIkernel(D)

% parameters
C = [0.1 1 10 100];
gamma = [0.25 0.5 1 2 4 8];
distPar = [1/16 1/8 0.5 1 2 4];

nFolds = 3;
tic
% matrix for saving results
AUC = zeros(length(distPar),length(gamma),length(C),nFolds);

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
    
    
    parfor d = 1:length(distPar)
        AUCtmp = zeros(length(gamma),length(C));
        
        for g = 1:length(gamma)
            
            opt = cell(1,2);
            opt{1} = distPar(d);
            opt{2} = gamma(g);
            
            for c = 1:length(C)
                
                model = trainMIkernelSVM(TRD,opt,C(c));
                % test the model
                [PL,SC,TL] = testMIkernelSVM(TED,model);
                perf = getClassifierPerfomance(PL,TL,SC);                AUCtmp(g,c) = perf.AUC;
                
                disp(['Dist = ' num2str(distPar(d)) ' C = ' num2str(C(c))...
                    ' gamma = ' num2str(gamma(g))...
                    '  AUC = ' num2str(AUCtmp(g,c))]);
            end
        end
        
        fn = ['temp/tmpValmig' num2str(distPar(d))];
        dlmwrite(fn,AUCtmp);
    end
    
    
    for d = 1:length(distPar)
        fn = ['temp/tmpValmig' num2str(distPar(d))];
        tmp = dlmread(fn);
        AUC(d,:,:,f) = tmp;
    end
end

AUC = mean(AUC,4);

% get the best performing parameters
[~,ind] = max(AUC(:));
[d,g,c] = ind2sub(size(AUC),ind);

disp(['the best config for MI-graph is : Dist = ' ...
    num2str(distPar(d)) ' C=' num2str(C(c)) ' and g=' num2str(gamma(g))])

C = C(c);
gamma = gamma(g);
distPar = distPar(d);
toc
end


function model = trainMIkernelSVM(D,opt,C)

W = milproxm(D,'miGraph',opt);
[K,Y] = getKernel(W,D);
optionStr = ['-s 0 -t 4 -c ' num2str(C) ' -q'];
model.SVM = svmtrain(Y,[(1:size(K,1))' K],optionStr);
model.kernel = W;

end

function [K,Y] = getKernel(kernelmap,D)

[~,nlab,bagid] = getbags(D);
nlab = ispositive(nlab);
Y = 2*nlab-1;

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


