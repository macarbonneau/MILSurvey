function [bagLabels, bagScores] = getBagLabels(D,iScores,iLabels)

[iScores] = correctDV(iScores,iLabels);

bagScores = zeros(length(D.B),1);
bagLabels = zeros(length(D.B),1);

for i = 1:length(D.B)
    
    idx = D.XtB == D.B(i);
    if sum(idx)>0
        bagScores(i) = max(iScores(idx));
        bagLabels(i) = max(iLabels(idx));
    else
        bagScores(i) = 0;
        bagLabels(i) = 0;
    end
end

%% print reports on accuracy
%disp(['Accuracy on bags: ' num2str(sum(bagLabels==D.YB)/length(D.YB)*100) ' %'])

end


function [DV] = correctDV(DV,PL)
sp = sum(DV(PL==1));
sn = sum(DV(PL~=1));

if sum(PL==1) >0
if sp < sn
    
    DV = -DV;
    
end
end
end