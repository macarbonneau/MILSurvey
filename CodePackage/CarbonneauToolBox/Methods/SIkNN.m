function out =  SIkNN(D,operation,model)


switch operation
    case 'train'
        
        % perform cross validation to find the best k
        out.k = CVSIkNN(D);
        out.X = D.X;
        out.Y = D.Y;
        
    case 'test'
        
        % give true labels
        out.TL = D.YR;
        out.TLB = D.YB;
        
        % classify update labels model
        SC = SIkNNClassify_mex(model.X,model.Y,D.X,model.k);
        out.SC = SC(:,model.k);
        out.PL = double(out.SC>=0.5);
          
        % get labels and score for bags
        [out.PLB, out.SCB] = getBagLabels(D,out.SC,out.PL);
end

end


function k = CVSIkNN(D)

nFolds = 5;
kmax = 21;


indexPerFold = listPerFold(D.Y,nFolds);

perf = zeros(nFolds,kmax);
for f = 1:nFolds
    
    [XT, YT, XV, YV] = getDatasetsForFold(D.X,D.Y,indexPerFold,f);
    [SC] = SIkNNClassify_mex(XT,YT,XV,kmax);
    
    for k = 1:kmax
        
        PL = double(SC(:,k)>=0.5);
        tmp = getClassifierPerfomance(PL,YV,SC(:,k));
        perf(f,k) = tmp.AUC;
    end
end

perf = mean(perf);
[~, k] = max(perf);


end


function [XTR, YTR, XVAL, YVAL] = getDatasetsForFold(X,Y,indexPerFold,valFoldNumber)
% Cette fonction retourne deux prdataset un pour la validation (VAL)
% et l'autre pour l'entrainement TR

% X : toutes les données dans une  matrice
% Y : les labels correspondant à X
% indexPerFold : est une structure de liste d'index obtenue avec la
% fonction obtenirListeParFold
% valFoldNumber: Est le numéro du fold utilisé pour la validation


idxTR = [];

for f = 1:size(indexPerFold,1)
    
    if f == valFoldNumber
        idxVal = indexPerFold{f};
    else
        idxTR = [idxTR;indexPerFold{f}];
    end
end

XTR = X(idxTR,:);
XVAL = X(idxVal,:);

YTR = Y(idxTR,:);
YVAL = Y(idxVal,:);

end

function [indexPerFold] = listPerFold(Y,nFolds)

% X est un prdataset
% nFolds est le nombre de folds désiré
% La fonction retourne une structure contenant des listes
% d'index. Il y a une liste d'index par fold.
indexPerFold = cell(nFolds,1);

idx = 1:size(Y,1);

% obtenir la liste de toutes les classes
Ynames = unique(Y);

ctr = 0;
for c = 1:length(Ynames)
    
    tmpI = idx(Y==Ynames(c));
    rp = randperm(length(tmpI));
    tmpI = tmpI(rp);
    
    for i = 1:length(tmpI)
        f = mod(ctr,nFolds)+1;
        indexPerFold{f} = [indexPerFold{f};tmpI(i)];
        ctr = ctr+1;
    end
    
end

end