function [TRD, TED] = getTrainingAndTestDatasets(fold,nFolds,BagPerFoldList,CD)

%% check if regular data set or MIL data set
if isa(CD,'MILdataset')
    [TRD, TED] = splitMIL(fold,nFolds,BagPerFoldList,CD);
else
    [TRD, TED] = splitReg(fold,nFolds,BagPerFoldList,CD);
end

if ~(performCheck(TRD))
    error('The TRD data set is not OK')
end
if ~(performCheck(TED))
    error('The TED data set is not OK')
end



end


function [TRD, TED] = splitMIL(fold,nFolds,BagPerFoldList,CD)

multiClass = length(unique(CD.Y))>2;
if ~multiClass
    posClass = max(unique(CD.Y));
    negClass = min(unique(CD.Y));
end

TRD = MILdataset;
TED = MILdataset;



% get bag list for both partitions
Btr = [];
Bte = [];
for i = 1:nFolds
    if fold ~= i
        Btr = [Btr;BagPerFoldList{i}];
    else
        Bte = [Bte;BagPerFoldList{i}];
    end
end

% get training data using obtained indexes
Xtr = []; Ytr = []; XtBtr = []; YBtr = []; YRtr = []; YStr = [];CBtr = [];
for i = 1:length(Btr)
    
    Xtr = [Xtr;CD.X(CD.XtB==Btr(i),:)];
    Ytr = [Ytr;CD.Y(CD.XtB==Btr(i),:)];
    XtBtr = [XtBtr;CD.XtB(CD.XtB==Btr(i),:)];
    if ~isempty(CD.YS)
        YStr = [YStr;CD.YS(CD.XtB==Btr(i),:)];
    end
    % get bag label
    if multiClass
        l = CD.Y(CD.XtB==Btr(i));
        % sanity check
        if length(unique(l))>1;error('PROBLEM WITH DATASET LABELS');end
        l = l(1);
    else
        bagIsPos = any(CD.Y(CD.XtB==Btr(i))==posClass);
        if bagIsPos
            l = posClass;
        else
            l = negClass;
        end
    end
    YBtr = [YBtr;l];
    
    if ~isempty(CD.CB)
        CBtr = [CBtr;CD.CB(CD.B==Btr(i),:)];
    end
    
    if ~isempty(CD.YR)
        YRtr = [YRtr;CD.YR(CD.XtB==Btr(i),:)];
    end
    
end

TRD.X = Xtr;
TRD.Y = Ytr;
TRD.XtB = XtBtr;
TRD.B = Btr;
TRD.YB = YBtr;
TRD.YR = YRtr;
TRD.YS = YStr;
TRD.CB = CBtr;

% get test data using obtained indexes
Xtest = []; Ytest = []; XtBtest = []; YBtest = []; YRtest = []; YStest = [];CBtest = [];
for i = 1:length(Bte)
    Xtest = [Xtest;CD.X(CD.XtB==Bte(i),:)];
    Ytest = [Ytest;CD.Y(CD.XtB==Bte(i),:)];
    XtBtest = [XtBtest;CD.XtB(CD.XtB==Bte(i),:)];
    if ~isempty(CD.YS)
        YStest = [YStest;CD.YS(CD.XtB==Bte(i),:)];
    end
    % get bag label
    if multiClass
        l = CD.Y(CD.XtB==Bte(i));
        % sanity check
        if length(unique(l))>1;error('PROBLEM WITH DATASET LABELS');end
        l = l(1);
    else
        bagIsPos = any(CD.Y(CD.XtB==Bte(i))==posClass);
        if bagIsPos
            l = posClass;
        else
            l = negClass;
        end
    end
    YBtest = [YBtest;l];
    
     if ~isempty(CD.CB)
        CBtest = [CBtest;CD.CB(CD.B==Bte(i),:)];
    end
    
    if ~isempty(CD.YR)
        YRtest = [YRtest;CD.YR(CD.XtB==Bte(i),:)];
    end
    
    
end

TED.X = Xtest;
TED.Y = Ytest;
TED.XtB = XtBtest;
TED.B = Bte;
TED.YB = YBtest;
TED.YR = YRtest;
TED.YS = YStest;
TED.CB = CBtest;
end

function [TRD, TED] = splitReg(fold,nFolds,BagPerFoldList,CD)
disp('Splitting a regular data set : NOT MIL')
% get bag list for both partitions
Ltr = [];
Lte = [];
for i = 1:nFolds
    if fold ~= i
        Ltr = [Ltr;BagPerFoldList{i}];
    else
        Lte = [Lte;BagPerFoldList{i}];
    end
end

TRD.X = CD.X(Ltr,:);
TRD.Y = CD.Y(Ltr);

TED.X = CD.X(Lte,:);
TED.Y = CD.Y(Lte);

end

function [ok] = performCheck(D)

ok = true;
   for b = 1:length(D.B)
       rl = sum(D.YR(D.XtB==D.B(b)));
       
       tmp = D.Y(D.XtB==D.B(b));
       for i = 2:length(tmp)    
            if tmp(i) ~= tmp(i-1)
               disp('PROBLEM WITH Y') 
               ok = false;
            end
       end
       
       if xor(rl > 0, sum(tmp) > 0)
          disp('PROBLEM with YR')
          ok = false;
       end
       
       if D.YB(b) ~= tmp(1) 
           disp('PROBLEM with YB') 
           ok = false;
       end
       
   end


end