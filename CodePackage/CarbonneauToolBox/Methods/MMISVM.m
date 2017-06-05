function out =  MMISVM(D,operation,model,opt)


switch operation
    case 'train'
        
        % D is the MIL dataset
        criterion = false;        
        iter = 0;
        
        %% initialize the selector vector from bag centroids
        Xtmp = zeros(length(D.B),size(D.X,2));
        
        for i = 1:length(D.B)    
            tmp = D.X(D.XtB==D.B(i),:);
            Xtmp(i,:) = mean(tmp);
        end
        
        % train the SVM with the centroids
        [model, ~, ~] = trainSVM_CV(Xtmp,D.YB,opt);
        [PL, ~, DV] = svmpredict(D.Y,D.X, model);
        
        % libSVM invert labels if the first label is neg
        [DV] = correctDV(DV,PL);
        
        % Find the most positive instance from all bags
        S = getSfromDecisionValue(DV,D);
        
        %% train the thing
        while ~criterion
            
            % save last selection vector
            Spast = S;
            
            % get logical index corresponding to selection
            logS = convertIndexToLogic(S, D);
            
            % train using the present labels
            %     [model, ~, ~] = trainSVMhere(D.X(logS,:),D.Y(logS));
            [model, ~, ~] = trainSVM_CV(D.X(logS,:),D.Y(logS),opt);
            
            % classify update labels using new model
            [out.PL, ~, DV] = svmpredict(D.Y,D.X, model);
            
            tmp =D.Y(logS);
            
            % invert decision value if negative score correspond to a
            % positive label
            [DV] = correctDV(DV,out.PL);
            
            % Find the most positive instance from all bags
            S = getSfromDecisionValue(DV,D);
            
            % verify if the algorithm converged
            if Spast == S
                disp(['converged correctly after ' num2str(iter) ' iterations'])
                criterion = true;
            end
            
            iter = iter +1;
            if opt.maxIter < iter
                criterion = true;
                disp('never converged mais je suis tanned')
            end
            
        end
        
        out = model;
        
    case 'test'
        
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;
        
        % classify update labels model
        [out.PL, ~, out.SC] = svmpredict(D.YR,D.X, model);
        
        % get labels and score for bags
        [out.PLB, out.SCB] = getBagLabels(D,out.SC,out.PL);
        
end

end


function S = getSfromDecisionValue(DV,D)
% Find the most positive instance from all bags
S = D.B*0;
for i = 1:length(D.B)
    if D.YB(i) == 1
        [~, S(i)] = max(DV(D.XtB==D.B(i)));
    end
end
end



function logS = convertIndexToLogic(S, D)

logS = false(size(D.Y));

for i = 1:length(D.B)
    
    if D.YB(i) == 1
        tmp = D.Y(D.XtB == D.B(i));
        tmp = logical(tmp*0);
        tmp(S(i)) = true;
        logS(D.XtB == D.B(i)) = tmp;
    else
        logS(D.XtB == D.B(i)) = true;
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

