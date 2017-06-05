function [AUC, pAUC] = getAUPR(scores, TL)
% scores:  are the scores for each instances
% TL are the true labels
percent = 0.3; %for computation of the pAUC

% make sure the vector are column vectors
if size(scores,1) > size(scores,2)
    scores = scores';
end
if size(TL,1) > size(TL,2)
    TL = TL';
end

% find all different threshold values
[scores, idx] = sort(scores,'ascend');
TL = TL(idx);
threshold = unique(scores);

if length(threshold) == 1
    AUC = 0;
    pAUC = 0;
    return
end




PR = zeros(size(threshold));
REC = PR;

% get  FPR and TPR for each threshold values
for t = 1:length(threshold)
    
    PL = scores > threshold(t);
    
    % Recall
    if sum(TL==1)> 0
        REC(t) = sum(PL==1 & TL==1)/sum(TL==1);
    else
        error('there are no positve cases in the data set')
    end
        
    % Precision
    PR(t) = sum(PL==1 & TL==1)/sum(PL==1);

end


idx = isnan(PR) | isnan(REC);
PR(idx) = [];
REC(idx) = [];
if isempty(PR)
   disp('OUPS PR EMPTY!!!!')
   PR = 0;
end


PR = fliplr(PR);
REC = fliplr(REC);

% add start and end points
PR = [PR(1) PR PR(end)];
REC = [0 REC 1];

% plot(REC,PR)

%% Compute AUC And pAUC
AUC = 0;
pAUC = 0;
for i = 1:length(REC)-1
    
    
    tmp = (PR(i+1)+PR(i))/2;
    tmp = tmp*(REC(i+1)-REC(i));
    
    %     tmp = (REC(i)+REC(i+1))/2*(PR(i+1)-PR(i));
    AUC = AUC + tmp;
    
    
    if PR(i) < percent
        if PR(i+1) <= percent % the interval in entirely in the partial area
            
            pAUC = pAUC + tmp;
            
        else % the interval in partially in the partial area
            
            Y1 = REC(i);
            Y2 = REC(i+1);
            X1 = PR(i);
            X2 = PR(i+1);
            m = (Y2-REC(i))/(X2-X1);
            Y3 = m*(percent-X1) + REC(i);
            tmp = (REC(i)+Y3)/2*(percent-PR(i));
            pAUC = pAUC + tmp;
            
        end
    end
    
end

% AUC1 = AUC;
% AUC = 0;

% for i = 1:length(REC)-1
%
%     tmp = ((REC(i) + REC(i+1))*.5)*(PR(i+1)-PR(i));
%     AUC = AUC + tmp;
%
%
% end



%% plot comparison

%
% subplot(2,1,1)
% plot(REC,PR)
% xlim([-.01, 1.01])
% ylim([-.01, 1.01])
% %
% subplot(2,1,2)
% % plot(X,Y)
% % xlim([-.01, 1.01])
% % ylim([-.01, 1.01])
% % title('VL')
% %
% % subplot(2,2,4)
% plot(X1M,Y1M)
% xlim([-.01, 1.01])
% ylim([-.01, 1.01])
% title('MAT')

% if AUC~=AUCM
%
%    disp(num2str(AUC))
%    disp(num2str(AUCM))
% %    error('PR CURVE')
% AUC=AUCM;
% end


end