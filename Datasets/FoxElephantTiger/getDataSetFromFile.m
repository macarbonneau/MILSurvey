function [] = getDataSetFromFile()

load('tiger')
D = normalizeUnitVarianceMIL(D);
save('TigerNormalized','D')
%%

load('fox')

D = normalizeUnitVarianceMIL(D);
save('FoxNormalized','D')

%%
load('elephant')

D = normalizeUnitVarianceMIL(D);
save('ElephantNormalized','D')




end


%% get Elephant Fox or Tiger
function [X,Y,XtB,YB,B] = getEleFoxTig(animal)

%fileName = ['Datasets/FoxElephantTiger/' animal];
load(animal)
X = x.data;
Y = x.nlab-1;
XtB = x.ident.milbag;

B = unique(x.ident.milbag);
YB = zeros(length(B),1);


for b = 1:length(B)
    ind = XtB == B(b);
    s = sum(Y(ind));
    if (s ~= length(Y(ind)) && s ~=0) %check for problem
        disp(['PROBLEM with the dataset ' animal])
        pause
    end
    
    if s > 0
        YB(b) = 1;
    end
end

end

%% get Synthetic Data
function [X,Y,XtB,YB,B] = getSynt()

A = dlmread('Datasets/Synthetic/SynDataset-NonUniform.data');

X = A(:,3:size(A,2));
Y = A(:,2);
XtB = A(:,1);

% get Bags
B = unique(XtB);
if size(B,1) < size(B,2)
    B = B';
end

% get Bag Labels
YB = zeros(size(B));
for b = 1:length(B)
    
    tmp = Y(XtB==b);
    if sum(tmp)>0
        YB(b) = 1;
    end
end

end

%% get Corel Dataset
function [X,Y,XtB,YB,B] = getCorel(numCat)

load('Datasets/Corel/imagefeatures.mat')

X = [];
Y = [];
B = 1:size(D,2);
B = B';
XtB = [];
YB = L;

for i = 1:size(D,2)
    
    tmpX = D{1,i}';
    X = [X;tmpX];
    tmpY = ones(size(tmpX,1),1)*L(i);
    Y = [Y;tmpY];
    tmpB = ones(size(tmpX,1),1)*i;
    XtB = [XtB;tmpB];
end

% Change all label to be between 1 and 20
Y = Y+1;
YB = YB+1;

if numCat == 100
    % remove ten last categories
    idx = Y<11;
    Y = Y(idx);
    X = X(idx,:);
    XtB = XtB(idx);
    idx = YB<11;
    B = B(idx);
    YB = YB(idx);
    
end

end


function [X,Y,XtB,YB,B] = getNewsgroup(fileCode)

load(['Datasets/Newsgroups/newsgroups' num2str(fileCode) '.mat'])

X = x.data;
Y = x.nlab-1;
XtB = x.ident.milbag;

B = unique(x.ident.milbag);
YB = zeros(length(B),1);


for b = 1:length(B)
    ind = XtB == B(b);
    s = sum(Y(ind));
    if (s ~= length(Y(ind)) && s ~=0) %check for problem
        disp(['PROBLEM with the dataset ' animal])
        pause
    end
    
    if s > 0
        YB(b) = 1;
    end
end


end




