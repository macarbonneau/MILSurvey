function [AUC, pAUC] = getAUROC(scores, labels, percent)


% [Y,X,INFO] = vl_roc(labels'-0.5,scores');
% X = 1-X;
%  [X1M,Y1M,~,AUCM] = perfcurve(labels,scores,1);
% AUCVL = INFO.auc

% make sure the vector are column vectors
if size(scores,1) > size(scores,2)
    scores = scores';
end
if size(labels,1) > size(labels,2)
    labels = labels';
end




[scores, idx] = sort(scores,'ascend');
labels = labels(idx);
threshold = unique(scores);
FPR = zeros(size(threshold));
TPR = FPR;

% get  FPR and TPR for each threshold values
for t = 1:length(threshold)
    
    PL = scores > threshold(t);
    
    TP = sum(double(PL>0 & labels>0));
    P = sum(double(labels>0));
    
    TPR(t) = TP/P;
    
    FP = sum(double(PL>0 & labels<=0));
    N = sum(double(labels<=0));
    
    if N ~= 0
    FPR(t) = FP/N;
    else
        FPR(t) = 0;
    end
end

% add start and end points
FPR = [1 FPR 0];
TPR = [1 TPR 0];

FPR = fliplr(FPR);
TPR = fliplr(TPR);


% plot(FPR,TPR)

%% Compute AUC And pAUC
AUC = 0;
pAUC = 0;
for i = 1:length(TPR)-1
    
    tmp = (TPR(i)+TPR(i+1))/2*(FPR(i+1)-FPR(i));
    AUC = AUC + tmp;
    
    
    if FPR(i) < percent     
        if FPR(i+1) <= percent % the interval in entirely in the partial area
            
            pAUC = pAUC + tmp;
            
        else % the interval in partially in the partial area
            
            Y1 = TPR(i);
            Y2 = TPR(i+1);
            X1 = FPR(i);
            X2 = FPR(i+1);
            m = (Y2-TPR(i))/(X2-X1);
            Y3 = m*(percent-X1) + TPR(i);
            tmp = (TPR(i)+Y3)/2*(percent-FPR(i));
            pAUC = pAUC + tmp;
            
        end
    end
    
end

AUC1 = AUC;
AUC = 0;

for i = 1:length(TPR)-1
   
    tmp = ((TPR(i) + TPR(i+1))*.5)*(FPR(i+1)-FPR(i));
    AUC = AUC + tmp; 
end

% AUCM
% AUC

%% plot comparison

% 
% subplot(2,2,1)
% plot(FPR,TPR)
% xlim([-.01, 1.01])
% ylim([-.01, 1.01])
% 
% subplot(2,2,3)
% plot(X,Y)
% xlim([-.01, 1.01])
% ylim([-.01, 1.01])
% title('VL')
% 
% subplot(2,2,4)
% plot(X1M,Y1M)
% xlim([-.01, 1.01])
% ylim([-.01, 1.01])
% title('MAT')



end

