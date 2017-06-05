function out =  SISVM(D,operation,model,opt)


switch operation
    case 'train'
        
        %train an SVM using cross validation
        out = trainSVM_CV(D.X,D.Y,opt);
 
    case 'test'
             
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;
        
        % classify update labels model
        [out.PL, ~, out.SC] = svmpredict(D.YR,D.X, model);
        [out.SC] = correctDV(out.SC,out.PL);
        
        % get labels and score for bags
        [out.PLB, out.SCB] = getBagLabels(D,out.SC,out.PL);        
end

end

function [DV] = correctDV(DV,PL)

% sometimes scores (decision values) are inverted with LIBSVM
sp = sum(DV(PL==1));
sn = sum(DV(PL~=1));

if sp < sn
    DV = -DV;
end

end
