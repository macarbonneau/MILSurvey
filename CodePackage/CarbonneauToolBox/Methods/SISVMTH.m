function out =  SISVMTH(D,operation,model,opt)


switch operation
    case 'train'
        
        out.TH = optimizeThreshold(D,opt);     
        out.SVM = trainSVM_CV(D.X,D.Y,opt);
 
    case 'test'
             
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;
        
        % classify update labels model
        [out.PL, ~, out.SC] = svmpredict(D.YR,D.X, model.SVM);
        
        % get labels and score for bags
        [out.SCB] = getScores_SIL(D,out.PL);  
        out.PLB = out.SCB > model.TH;
end

end

function TH = optimizeThreshold(D,opt)

nFolds = 5;

BagPerFoldList = divideBagsInFolds(nFolds,D);

maxTH = zeros(nFolds,1);
minTH = zeros(nFolds,1);

for fold = 1:nFolds
        
    %% create training and test datasets
    [TRD, TED] = getTrainingAndTestDatasets(fold,nFolds,BagPerFoldList,D);
    
    % train a SVM
    model = trainSVM_CV(TRD.X,TRD.Y,opt);
    
    % classify validation data
    PL = svmpredict(TED.YR,TED.X, model);
    
    % get bag scores
    [BSC] = getScores_SIL(TED,PL);
    
    TH = unique(BSC);
    TH = sort(TH);
    bestAcc = 0;
    for i = 1:length(TH)
        PLB = BSC>TH(i);
        acc = mean(TED.YB == PLB);
        if acc > bestAcc
            bestAcc = acc;
            maxTH(fold) = TH(i);
            minTH(fold) = TH(i);
        elseif acc == bestAcc
            maxTH(fold) = TH(i);
        end
        
    end
    
end

TH = (minTH+maxTH)/2;
TH = mean(TH);

end


function [bagScores] = getScores_SIL(D,iLabels)

bagScores = zeros(length(D.B),1);

for i = 1:length(D.B)
    
    idx = D.XtB == D.B(i);
    if sum(idx)>0
        bagScores(i) = mean(iLabels(idx));
       
    else
        bagScores(i) = 0;
    end
end

end