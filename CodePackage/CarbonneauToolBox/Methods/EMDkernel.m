function out =  EMDkernel2(D,DT,operation,model,opt)


switch operation
    case 'train'
        
        % get the model
        out = trainMIkernelSVM(D,opt);
        
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


function model = trainMIkernelSVM(D,opt)

% Embed bags
D = computeEMDKernel(D,[]);
% train SVM
[model.SVM] = trainSVM_CV(D.CB,D.YB,opt);

end

function [PL,SC,TL] = testMIkernelSVM(D,DT,model)

% Embedd bags
[D,DT] = computeEMDKernel(D,DT);

% classify with the SVM
[PL, ~, SC] = svmpredict(DT.YB, DT.CB, model.SVM);
TL = DT.YB;

end


function [TRD,TED] = computeEMDKernel(TRD,TED)


%% Encode all bags in the training set
NTR = size(TRD.B,1);
K = zeros(NTR);
for i = 1:NTR
    for j = 1:NTR
        
        Bi = TRD.X(TRD.XtB==TRD.B(i),:);
        Bj = TRD.X(TRD.XtB==TRD.B(j),:);
        ni = size(Bi,1);
        nj = size(Bj,1);
        
        DM = zeros(ni,nj);
        
        for k = 1:ni
            for l = 1:nj
                DM(k,l) = sqrt(sum((Bi(k,:)-Bj(l,:)).^2));
            end
        end
        K(i,j) = emd_mex(ones(1,ni)/ni,ones(1,nj)/nj,DM);
        
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
            
            ni = size(Bi,1);
            nj = size(Bj,1);
            
            DM = zeros(ni,nj);
            
            for k = 1:ni
                for l = 1:nj
                    DM(k,l) = sqrt(sum((Bi(k,:)-Bj(l,:)).^2));
                end
            end
            KT(i,j) = emd_mex(ones(1,ni)/ni,ones(1,nj)/nj,DM);
            
        end
    end
    
    TED.CB = KT;
    
end


end


