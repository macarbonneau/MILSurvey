function [out] = getClassifierPerfomance(PL,TL,SC)
% PL:   predicted labels
% TL:   true labels
% SC:   score
% -----------------------------------------------
% ACC:  Accuracy
% REC:  Recall
% PRE:  Precision
% FPR:  False positive

% convert labels to logical
PL = PL > 0;
TL = TL > 0;

% verify that there is only one comlumns in score (SC)
if size(SC,2) > size(SC,1)
   SC = SC'; 
end
if size(SC,2) > 1
   SC = SC(:,1); 
end
    

% verify that positive score correspond to positive labels
if sum(SC(PL))/length(SC(PL)) < sum(SC(~PL))/length(SC(~PL))
SC = -SC;
end


% Accuracy
out.ACC = sum(~xor(PL,TL))/length(PL);

% Recall
if sum(TL==1)> 0
    out.REC = sum(PL==1 & TL==1)/sum(TL==1);
else
    error('there are no positve cases in the data set')
end


% Precision
if sum(PL==1)>0
    out.PRE = sum(PL==1 & TL==1)/sum(PL==1);
else
    out.PRE = 1;
end

% False Positive Rate
if sum(PL==1)>0
    out.FPR = sum(PL==1 & TL==0)/sum(PL==1);
else
    out.FPR = 1;
end

% F1 score
out.F1 = 2*out.PRE*out.REC/(out.PRE+out.REC);
if isnan(out.F1)
out.F1 = 0;

end




% AUC ROC
[out.AUC, out.pAUC] = getAUROC(SC, TL, .30);
[out.AUCPR] = getAUPR(SC, TL);

% unwweigthed average recall
out.UAR = 0.5*sum(PL==1 & TL==1)/sum(TL==1) + 0.5*sum(PL==0 & TL==0)/sum(TL==0);


end