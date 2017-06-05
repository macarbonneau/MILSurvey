function BagPerFoldList = divideBagsInFolds(nFolds,dataset)

if length(unique(dataset.YB)) == 2
    BagPerFoldList = for2Class(nFolds,dataset);
else
    BagPerFoldList = forMultiClass(nFolds,dataset);
end

end



function BagPerFoldList = for2Class(nFolds,dataset)

BagPerFoldList = cell(nFolds,1);

% get list of positive and negative bags
pi = dataset.YB == max(unique(dataset.YB));
pBi = dataset.B(pi);
nBi = dataset.B(~pi);

% shuffle the lists
ri = randperm(length(pBi));
pBi = pBi(ri);
ri = randperm(length(nBi));
nBi = nBi(ri);

% add positive bags to fold lists
nbpf = floor(length(pBi)/nFolds); % min number of bag per fold
for i = 1:nFolds
    
    BagPerFoldList{i} = pBi(1:nbpf);
    pBi(1:nbpf) = [];
end

% add remaining bags to folds
for i = 1:length(pBi)
    
    BagPerFoldList{i} = [BagPerFoldList{i}; pBi(i)];
end

% add negative bags to fold lists
nbpf = floor(length(nBi)/nFolds); % min number of bag per fold
for i = 1:nFolds
    
    BagPerFoldList{i} = [BagPerFoldList{i}; nBi(1:nbpf)];
    nBi(1:nbpf) = [];
end

% add remaining bags to folds
for i = 1:length(nBi)
    
    BagPerFoldList{i} = [BagPerFoldList{i}; nBi(i)];
end

end



function BagPerFoldList = forMultiClass(nFolds,dataset)

BagPerFoldList = cell(nFolds,1);

ClassList = unique(dataset.YB);
nC = length(ClassList);

% Add each class individually
for c = 1:nC
    
    % get list of bags for this class
    pi = dataset.YB == ClassList(c);
    pBi = dataset.B(pi);
    
    % shuffle the lists
    ri = randperm(length(pBi));
    pBi = pBi(ri);
    
    % add bags to fold lists
    nbpf = floor(length(pBi)/nFolds); % min number of bag per fold
    for i = 1:nFolds
        
        BagPerFoldList{i} = [BagPerFoldList{i}; pBi(1:nbpf)];
        pBi(1:nbpf) = [];
    end
    
    % add remaining bags to folds
    for i = 1:length(pBi)        
        BagPerFoldList{i} = [BagPerFoldList{i}; pBi(i)];
    end
    
    
end

end