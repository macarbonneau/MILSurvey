function [TRD, TED] = normalizeUnitVarianceMIL(TRD, TED)



% X is a dataset with row entries
u = mean(TRD.X);
s = std(TRD.X)+eps;
um = repmat(u,size(TRD.X,1),1);
sm = repmat(s,size(TRD.X,1),1);
TRD.X = (TRD.X-um)./sm;

if nargin == 2
um = repmat(u,size(TED.X,1),1);
sm = repmat(s,size(TED.X,1),1);
TED.X = (TED.X-um)./sm;
else
    TED = [];
end


% verification
% mean(TRD.X)
% std(TRD.X)
% mean(TED.X)
% std(TED.X)

end