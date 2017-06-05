function out =  MInD(D,DT,operation,model,opt)

switch operation
    case 'train'
        
        % get the model
        out = trainMIkernelSVM(D,DT,opt);
        
    case 'test'
        
        % give true labels
        out.TL = DT.YR;
        out.PL = out.TL;
        out.SC = out.TL;
        out.TLB = DT.YB;
        
        % classify instances and bags
        [out.PLB,out.SCB,out.TLB] = testMIkernelSVM(D,DT,model);
end

end


function model = trainMIkernelSVM(D,DT,opt)

% Embed bags
[D,DT] = meanMinKernel(D,DT);
% train SVM
[model.SVM] = trainSVM_CV(D.CB,D.YB,opt);

end


function [PL,SC,TL] = testMIkernelSVM(D,DT,model)

% Embedd bags
[D,DT] = meanMinKernel(D,DT);

% classify with the SVM
[PL, ~, SC] = svmpredict(DT.YB, DT.CB, model.SVM);
TL = DT.YB;

end

function [TRD,TED] = meanMinKernel(TRD,TED)


%% Encode all bags in the training set
NTR = size(TRD.B,1);
K = zeros(NTR);
for i = 1:NTR
    for j = 1:NTR
        Bi = TRD.X(TRD.XtB==TRD.B(i),:);
        Bj = TRD.X(TRD.XtB==TRD.B(j),:);
        
        allDforMean = zeros(size(Bi,1),1);
        for k = 1:size(Bi,1)
            allDforMin = zeros(size(Bj,1),1);
            for l = 1:size(Bj,1)
                allDforMin(l) = sum((Bi(k,:)-Bj(l,:)).^2);
            end
            allDforMean(k) = min(allDforMin);
        end
        
        K(i,j) = mean(allDforMean);
        
    end
end
TRD.CB = K;

%% Encode all bags in the test set
if ~isempty(TED)
    NTE = size(TED.B,1);
    KT = zeros(NTE,NTR);
    
    for i = 1:NTE
        for j = 1:NTR
            
            Bi = TED.X(TED.XtB==TED.B(i),:);
            Bj = TRD.X(TRD.XtB==TRD.B(j),:);
            
            allDforMean = zeros(size(Bi,1),1);
            for k = 1:size(Bi,1)
                allDforMin = zeros(size(Bj,1),1);
                for l = 1:size(Bj,1)
                    allDforMin(l) = sum((Bi(k,:)-Bj(l,:)).^2);
                end
                allDforMean(k) = min(allDforMin);
            end
            
            KT(i,j) = mean(allDforMean);
            
        end
    end
    
    TED.CB = KT;
    
end
end

