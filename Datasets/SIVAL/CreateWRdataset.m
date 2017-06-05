function [] = CreateWRdataset()

clear all
close all
clc
WR = 0.05:0.05:0.5;

allDataset = dir('*.data');

for d = 1:length(allDataset)
    
    disp(['TREATING: ' allDataset(d).name])
    
    fid = fopen(allDataset(d).name);
    Dtot = getDataFromFile(fid);
    fclose(fid);
        
end

end


function [D] = getDataFromFile(fid)


D = MILdataset;

frmt = ['%s ' repmat('%f ',1,32)];

tmp = textscan(fid,frmt,'CommentStyle','%','CollectOutput',true,'delimiter',',');
XtBText = tmp{1};
BagNames = unique(XtBText);
tmp = tmp{2};
D.X = tmp(:,2:31);
D.YR = tmp(:,32);
D.XtB = ones(size(D.X,1),1);
D.B = 1:length(BagNames);
D.B = D.B';
IN = tmp(:,1);
b = 1;


for i = 2:size(D.X,1)
   
    if IN(i-1) > IN(i) % new bag detected
        
        b = b+1;
    end
 
    D.XtB(i) = b;   
end

maxWr = 0;
minWr = 100;
for b = 1:length(D.B)
    D.YB(b) = double(sum(D.YR(D.B(b)==D.XtB))>0);
    wr = sum(D.YR(D.B(b)==D.XtB))/length(D.YR(D.B(b)==D.XtB))*100;
    if wr ~= 0
    if wr > maxWr
       maxWr = wr; 
    elseif wr < minWr
        minWr = wr;
    end
    end
end

disp(['Minimum witness rate is ' num2str(minWr)])
disp(['Maximum witness rate is ' num2str(maxWr)])

end


function O = adjustWitnessRate(D,WR)

% number of negative instances in the bags
NN =(1/WR);

O = MILdataset;
O.B = D.B;
O.YB = D.YB;

for b = 1:length(D.B)
    
    
    % creation of postive bags
    if D.YB(b) == 1
        
        % the positive instance
        idxp = D.XtB==D.B(b) & D.YR;
        tmpX = D.X(idxp,:);
        tmpX = tmpX(1,:);
        tmpY = 1;
        
 
        % the negative instances
        idxn = D.XtB==D.B(b) & ~logical(D.YR);
        tmp = D.X(idxn,:);
        
        NI = ceil(min(NN-1,size(tmp,1)));
    
        
        tmpX = [tmpX;tmp(1:NI,:)];
        tmp = D.YR(idxn);
        tmpY = [tmpY;tmp(1:NI)];
        
        O.X = [O.X;tmpX];
        O.Y = [O.Y;ones(size(tmpX,1),1)*D.YB(b)];
        O.YR = [O.YR;tmpY];
        O.XtB = [O.XtB;ones(size(tmpX,1),1)*D.B(b)];
    else
        
        % the negative instances
        idxn = D.XtB==D.B(b);
        idxn = idxn & (D.YR ~= 1);
        tmp = D.X(idxn,:);
        
        NI = ceil(min(NN,size(tmp,1)));
    
        tmpX = tmp(1:NI,:);
        tmp = D.YR(idxn);
        tmpY = tmp(1:NI);
        
        O.X = [O.X;tmpX];
        O.Y = [O.Y;ones(size(tmpX,1),1)*D.YB(b)];
        O.YR = [O.YR;tmpY];
        O.XtB = [O.XtB;ones(size(tmpX,1),1)*D.B(b)];
        
        
    end
    
end


end
